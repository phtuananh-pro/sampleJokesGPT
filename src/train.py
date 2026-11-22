"""Train the joke chatbot's decoder-only Transformer from serialized text.

The loop samples token windows, predicts the token one position ahead,
computes cross-entropy, and updates the model with AdamW. Run from the
repository root with ``python -m src.train``.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import JokeTokenizer
from .training_data import encode_conversations, parse_conversations, sample_batch, split_by_joke


# Smaller defaults keep the CPU path usable for a teaching-sized run.
CPU_DEFAULTS = {
    "bpe_vocab_size": 2000,
    "bpe_min_pair_freq": 2,
    "bpe_train_chars": 3_000_000,
    "block_size": 256,
    "batch_size": 32,
    "max_iters": 5000,
    "eval_interval": 100,
    "eval_iters": 100,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "dropout": 0.2,
}

# CUDA defaults use more parallel work and a wider/deeper model.
GPU_DEFAULTS = {
    "bpe_vocab_size": 2000,
    "bpe_min_pair_freq": 2,
    "bpe_train_chars": 3_000_000,
    "block_size": 128,
    "batch_size": 64,
    "max_iters": 10000,
    "eval_interval": 500,
    "eval_iters": 100,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.2,
}


def choose_device(requested: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU.

    Explicit CUDA requests fail clearly if this PyTorch installation cannot
    access an NVIDIA GPU, matching the behavior of the sampleGPT companion.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return requested


def confirm_overwrite(out_dir: Path) -> bool:
    """Ask before replacing an existing run and its saved checkpoint."""
    while True:
        try:
            answer = input(f"Output folder already exists: {out_dir}\nOverwrite it? [y/N]: ").strip().lower()
        except EOFError:
            print("No confirmation received. Exiting without changing the existing run.")
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            print("Keeping the existing run. Exiting.")
            return False
        print("Please answer yes or no.")


def main() -> None:
    """Prepare text, train/evaluate the model, and save its best checkpoint."""
    args = parse_train_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    runtime_info = get_runtime_info(args.device)
    log(f"Runtime info: {runtime_info}")

    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if out_dir.is_symlink():
        raise RuntimeError(f"Refusing to manage an output folder through a symbolic link: {out_dir}")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"Output path exists and is not a folder: {out_dir}")
        if not confirm_overwrite(out_dir):
            return
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    text = data_path.read_text(encoding="utf-8")
    all_examples = parse_conversations(text)
    train_examples, val_examples = split_by_joke(all_examples, args.seed)
    log(f"Parsed {len(all_examples):,} complete chat samples; split by unique jokes")
    log("Training byte-level BPE tokenizer...")
    # Training on a prefix speeds tokenizer construction; the model still sees
    # the full encoded dataset below.
    if args.bpe_train_chars > 0:
        tokenizer_train_path = out_dir / "_tokenizer_train_sample.txt"
        tokenizer_train_path.write_text(text[: args.bpe_train_chars], encoding="utf-8")
    else:
        tokenizer_train_path = data_path

    tokenizer = JokeTokenizer.train_from_file(
        tokenizer_train_path,
        vocab_size=args.bpe_vocab_size,
        min_frequency=args.bpe_min_pair_freq,
        show_progress=True,
    )
    tokenizer.save(out_dir / "tokenizer.json")
    if tokenizer_train_path.parent == out_dir:
        tokenizer_train_path.unlink(missing_ok=True)

    log("Encoding complete training/validation conversations...")
    train_data, skipped_train = encode_conversations(train_examples, tokenizer, args.block_size)
    val_data, skipped_val = encode_conversations(val_examples, tokenizer, args.block_size)
    if not train_data or not val_data:
        raise ValueError(
            "No complete examples fit the context window in one split. "
            "Increase --block_size or shorten the corpus examples."
        )
    log(
        f"Usable conversations: train={len(train_data):,}, val={len(val_data):,}; "
        f"skipped too long: train={skipped_train:,}, val={skipped_val:,}"
    )

    # Keep a copy with the run so the checkpoint remains interpretable.
    shutil.copyfile(data_path, out_dir / "data_clean.txt")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_step = None
    log(f"Device: {args.device}")
    pad_id = tokenizer.tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Tokenizer does not contain the padding token")
    log(
        f"Model configuration: parameters={sum(p.numel() for p in model.parameters()):,}; "
        f"vocab={config.vocab_size:,}; context={config.block_size}; "
        f"layers={config.n_layer}; heads={config.n_head}; embedding={config.n_embd}; "
        f"dropout={config.dropout}; device={args.device}"
    )
    log(f"BPE vocabulary: {tokenizer.vocab_size:,}")
    log(f"Train / validation conversations: {len(train_data):,} / {len(val_data):,}")

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model, train_data, val_data, args.batch_size, args.block_size,
                pad_id, args.device, args.eval_iters,
            )
            log(
                f"step {step}/{args.max_iters}: train loss {losses['train']:.4f}, "
                f"validation loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_step = step
                save_checkpoint(out_dir, model, config, args, best_val_loss)

        # Each sampled x window is paired with y shifted one token to the right.
        x, y = sample_batch(train_data, args.batch_size, args.block_size, pad_id, args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    save_config_to_file(
        args, data_path, out_dir, runtime_info, text, tokenizer,
        train_data, val_data, skipped_train, skipped_val, config, best_val_loss, best_step,
    )
    log(f"Training complete. Best validation loss: {best_val_loss:.4f} at step {best_step}.")
    log(
        f"Saved model configuration: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"{config.n_layer} layers, {config.n_head} heads, {config.n_embd} embedding width, "
        f"{config.block_size} context, {config.vocab_size} vocabulary, "
        f"dropout={config.dropout}; checkpoint step={best_step}"
    )
    log(f"Saved model, tokenizer, config, and data copy to: {out_dir}")


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, pad_id, device, eval_iters):
    """Evaluate assistant-only cross-entropy on complete held-out jokes."""
    model.eval()
    result = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = sample_batch(data, batch_size, block_size, pad_id, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        result[split] = losses.mean().item()
    model.train()
    return result


def save_checkpoint(out_dir: Path, model: GPT, config: GPTConfig, args, best_val_loss: float) -> None:
    """Save learned tensors plus enough configuration to reconstruct GPT."""
    torch.save(model.state_dict(), out_dir / "model.pt")
    payload = {
        "model": asdict(config),
        "training": vars(args),
        "best_val_loss": best_val_loss,
    }
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_config_to_file(
    args, data_path, out_dir, runtime_info, text, tokenizer,
    train_data, val_data, skipped_train, skipped_val, config, best_val_loss, best_step,
) -> None:
    """Write final provenance and training statistics to config.json."""
    dataset_stats = {
        "raw_chars": len(text),
        "total_conversations": len(train_data) + len(val_data) + skipped_train + skipped_val,
        "train_conversations": len(train_data),
        "val_conversations": len(val_data),
        "train_skipped_over_context": skipped_train,
        "val_skipped_over_context": skipped_val,
        "split_strategy": "by_distinct_joke_before_prompt_augmentation",
        "loss_scope": "assistant_tokens_including_eos",
    }
    payload = build_config_payload(
        args, config, tokenizer, data_path, out_dir, dataset_stats,
        runtime_info, best_val_loss, best_step, status="finished",
    )
    save_config(out_dir, payload)


def build_config_payload(
    args, model_config: GPTConfig, tokenizer: JokeTokenizer,
    data_path: Path, out_dir: Path, dataset_stats: dict, runtime_info: dict,
    best_val_loss: float | None = None, best_step: int | None = None,
    status: str = "initialized",
) -> dict:
    """Create the structured run record read by chat and useful for auditing."""
    return {
        "status": status,
        "paths": {
            "data": str(data_path), "out_dir": str(out_dir),
            "model_file": str(out_dir / "model.pt"),
            "tokenizer_file": str(out_dir / "tokenizer.json"),
            "config_file": str(out_dir / "config.json"),
            "copied_clean_data_file": str(out_dir / "data_clean.txt"),
        },
        "model": asdict(model_config),
        "tokenizer": {
            "type": "huggingface_tokenizers_byte_level_bpe",
            "vocab_size": tokenizer.vocab_size,
            "requested_vocab_size": args.bpe_vocab_size,
            "min_pair_freq": args.bpe_min_pair_freq,
            "train_chars": args.bpe_train_chars,
            "special_tokens": tokenizer.SPECIAL_TOKENS,
        },
        "training": {
            key: getattr(args, key) for key in (
                "data", "out_dir", "block_size", "batch_size", "max_iters",
                "eval_interval", "eval_iters", "learning_rate", "weight_decay",
                "grad_clip", "seed", "device", "n_layer", "n_head", "n_embd",
                "dropout", "bpe_vocab_size", "bpe_min_pair_freq", "bpe_train_chars",
            )
        },
        "dataset": dataset_stats,
        "checkpoint": {"best_val_loss": best_val_loss, "best_step": best_step},
        "runtime": runtime_info,
    }


def save_config(out_dir: Path, payload: dict) -> None:
    """Write the complete run record as readable UTF-8 JSON."""
    (out_dir / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_runtime_info(device: str) -> dict:
    """Capture Python, platform, PyTorch, and available NVIDIA device details."""
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_count"] = torch.cuda.device_count()
    return info


def normalize_device(device: str) -> str:
    """Resolve the ``gpu`` alias and apply the common auto/CUDA selection."""
    return choose_device("cuda" if device.lower() == "gpu" else device.lower())


def apply_device_defaults(args):
    """Fill unset hyperparameters from the selected CPU or CUDA preset."""
    defaults = GPU_DEFAULTS if args.device == "cuda" else CPU_DEFAULTS
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_arg_parser() -> argparse.ArgumentParser:
    """Define the data, device, tokenizer, model, and optimizer options."""
    parser = argparse.ArgumentParser(description="Train the sampleJokesGPT chatbot.")
    parser.add_argument("--data", default="data/clean_data.txt")
    parser.add_argument("--out_dir", default="models/jokes_bpe")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "gpu"],
        help="auto selects CUDA when available, otherwise CPU; gpu aliases cuda",
    )
    parser.add_argument("--bpe_vocab_size", type=int, default=None)
    parser.add_argument("--bpe_min_pair_freq", type=int, default=None)
    parser.add_argument("--bpe_train_chars", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def parse_train_args():
    """Parse command-line options and resolve device-specific defaults."""
    args = build_arg_parser().parse_args()
    args.device = normalize_device(args.device)
    return apply_device_defaults(args)


def log(message: str) -> None:
    """Print one concise training status line with a local timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


def get_device() -> str:
    """Return CUDA if PyTorch can use it, or CPU otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
