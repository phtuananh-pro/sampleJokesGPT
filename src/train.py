"""Teach a decoder-only GPT to complete *one whole joke* from a request.

Educational changes from random text-window training:
  1. each supervised sequence is one bounded conversation, never two jokes;
  2. a learned joke-ID token anchors one specific joke in model parameters;
  3. prompt/padding targets are masked (loss only on joke ID, text and EOS);
  4. record-level validation keeps the same joke out of both splits.

Every target still comes from the NEXT token in the source sequence. Joke IDs
are part of the generative model's internal vocabulary, not external retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT, GPTConfig
from .tokenizer import JokeTokenizer
from .topics import topic_prompt


def log(message: str, *, step: int | None = None, total: int | None = None) -> None:
    """Timestamp console messages, including completed/total training steps."""
    progress = f" [{step}/{total}]" if step is not None and total is not None else ""
    print(f"[{datetime.now().strftime('%H:%M:%S')}]{progress} {message}", flush=True)


def format_duration(seconds: float | None) -> str:
    """Show elapsed time and ETA as hours:minutes:seconds."""
    if seconds is None:
        return "n/a"
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def model_summary(model: GPT, device: str) -> str:
    """Describe architecture, hardware, and trainable parameter count."""
    cfg = model.config
    params = sum(p.numel() for p in model.parameters())
    processor = torch.cuda.get_device_name(0) if device == "cuda" else (platform.processor() or platform.machine())
    return (
        f"device={device} ({processor}); parameters={params:,} ({params / 1e6:.2f}M); "
        f"layers={cfg.n_layer}; heads={cfg.n_head}; embedding={cfg.n_embd}; "
        f"context={cfg.block_size}; vocabulary={cfg.vocab_size}; dropout={cfg.dropout}"
    )


STYLE_PROMPTS = {
    "dad_joke": "Tell me a dad joke.",
    "pun": "Tell me a pun.",
    "one_liner": "Tell me a one-liner.",
    "wordplay": "Tell me a wordplay joke.",
}


def load_records(path: Path) -> list[dict]:
    """Read pre-curated JSONL, validating its four teaching-data fields."""
    records = []
    for line_num, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"Invalid JSONL at line {line_num}: {ex}") from ex
        if not isinstance(obj.get("joke"), str) or not obj["joke"].strip():
            raise ValueError(f"Line {line_num} is missing its joke text")
        if not isinstance(obj.get("topics"), list) or not isinstance(obj.get("style"), str):
            raise ValueError(f"Line {line_num} lacks topics or style metadata")
        records.append(obj)
    if len(records) < 20:
        raise ValueError("Expected at least 20 curated jokes for train/validation")
    return records


def canonical_prefix(prompt: str) -> str:
    """A user prompt whose following assistant text will be generated."""
    return f"<|bos|><|user|>{prompt}<|assistant|>"


def joke_id(n: int) -> str:
    """Atomic identity token learned alongside each joke (not a lookup key)."""
    return f"<|j{n:04d}|>"


def examples_for_record(record: dict, index: int, supported: set[str]):
    """Pair the SAME full answer with generic, style, and topic prompts.

    If the same joke were present under unrelated answers, the model could
    blend its punchlines. Keeping one identity across prompts helps it learn a
    stable continuation from the identifier it predicts first.
    """
    prompts = ["Tell me a joke."]
    if record["style"] in STYLE_PROMPTS:
        prompts.append(STYLE_PROMPTS[record["style"]])
    for topic in record["topics"]:
        if topic in supported:
            prompts.append(topic_prompt(topic))
    suffix = joke_id(index) + record["joke"].strip() + "<|eos|>"
    for prompt in dict.fromkeys(prompts):
        yield canonical_prefix(prompt), suffix


def prepare_examples(records: list[dict], tokenizer: JokeTokenizer, block_size: int,
                     supported: set[str]) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Create fixed-length x/y tensors with loss only on assistant completion.

    Targets use -100 for padding and user prompt positions, which PyTorch's
    cross-entropy ignores. This prevents the model optimizing for repeating
    the user prompt instead of producing complete jokes.
    """
    examples, skipped = [], 0
    pad_id = tokenizer.token_id("<|pad|>")
    for index, row in records:
        for prefix, suffix in examples_for_record(row, index, supported):
            full_ids = tokenizer.encode(prefix + suffix)
            prefix_count = len(tokenizer.encode(prefix))
            if len(full_ids) > block_size + 1:
                skipped += 1
                continue
            x_ids = full_ids[:-1]
            y_ids = full_ids[1:]
            for k in range(max(0, prefix_count - 1)):
                y_ids[k] = -100
            n_pad = block_size - len(x_ids)
            x_ids.extend([pad_id] * n_pad)
            y_ids.extend([-100] * n_pad)
            assert len(x_ids) == len(y_ids) == block_size
            assert any(y >= 0 for y in y_ids)
            examples.append((torch.tensor(x_ids), torch.tensor(y_ids)))
    return examples, skipped


def batch_from_examples(samples, batch_size: int, device: str):
    """Sample complete joke conversations; never cross sample boundaries."""
    indices = torch.randint(0, len(samples), (batch_size,)).tolist()
    x = torch.stack([samples[i][0] for i in indices]).to(device)
    y = torch.stack([samples[i][1] for i in indices]).to(device)
    return x, y


def completion_loss(model: GPT, x: torch.Tensor, y: torch.Tensor,
                    id_loss_weight: float = 5.0) -> torch.Tensor:
    """Weighted next-token loss on joke identity, body, and EOS.

    The first unmasked target is the joke identity. Giving it more weight
    helps the model learn the topic-to-identity association instead of
    spending nearly all gradient signal on common joke-body words.
    """
    logits, _ = model(x)
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
        ignore_index=-100, reduction="none",
    ).view_as(y)
    valid = y != -100
    first_completion = valid & (valid.long().cumsum(dim=1) == 1)
    weights = valid.float() + (id_loss_weight - 1.0) * first_completion.float()
    return (per_token * weights).sum() / weights.sum().clamp_min(1)



@torch.no_grad()
def estimate_loss(model: GPT, samples, steps: int, batch_size: int, device: str, id_loss_weight: float):
    model.eval()
    numbers = []
    for _ in range(steps):
        x, y = batch_from_examples(samples, batch_size, device)
        numbers.append(completion_loss(model, x, y, id_loss_weight).item())
    model.train()
    return sum(numbers) / len(numbers)


def choose_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_args():
    p = argparse.ArgumentParser(description="Train a topic-conditioned joke GPT from scratch")
    p.add_argument("--data", default="data/jokes_clean.jsonl")
    p.add_argument("--topics", default="data/topics.json")
    p.add_argument("--out_dir", default="models/jokes_v2")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bpe_vocab_size", type=int, default=4500)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--id_loss_weight", type=float, default=5.0)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=8)
    p.add_argument("--min_topic_examples", type=int, default=10)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    args.device = choose_device(args.device)
    if args.n_layer is None:
        args.n_layer = 5 if args.device == "cuda" else 3
    if args.n_embd is None:
        args.n_embd = 320 if args.device == "cuda" else 192
    if args.batch_size is None:
        args.batch_size = 24 if args.device == "cuda" else 8
    if args.max_iters is None:
        args.max_iters = 12000 if args.device == "cuda" else 3000
    if args.id_loss_weight < 1:
        p.error("id_loss_weight must be at least 1")
    if args.n_embd % args.n_head:
        p.error("n_embd must be divisible by n_head")
    if not (0 < args.val_fraction < 0.25):
        p.error("val_fraction must be between 0 and 0.25")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    source = Path(args.data)
    records = load_records(source)
    topics_config = json.loads(Path(args.topics).read_text(encoding="utf-8"))
    counts = Counter(t for r in records for t in set(r["topics"]))
    supported = {name for name, cnt in counts.items()
                 if cnt >= args.min_topic_examples
                 and name in topics_config["topic_keyword_patterns"]}
    rng = random.Random(args.seed)
    indexed = list(enumerate(records))
    rng.shuffle(indexed)
    cut = max(1, int(len(indexed) * args.val_fraction))
    validation_rows, train_rows = indexed[:cut], indexed[cut:]
    train_ids = [joke_id(idx) for idx, _ in train_rows]
    val_ids = [joke_id(idx) for idx, _ in validation_rows]

    out = Path(args.out_dir)
    if out.exists():
        if not args.overwrite:
            response = input(f"Replace existing model folder {out}? [y/N]: ").strip().lower()
            if response not in {"y", "yes"}:
                print("Existing checkpoint left unchanged")
                return
        if out.is_symlink() or not out.is_dir():
            raise RuntimeError("Refusing to delete a symlink or a non-directory output")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    log(f"Source: {len(records):,} curated jokes; supported topics={len(supported)}; "
        f"train/validation records={len(train_rows)}/{len(validation_rows)}; "
        f"optimization steps={args.max_iters}", step=0, total=args.max_iters)
    # Train tokenizer on the prompts and jokes used by this run. Validation
    # records are allowed to influence BPE segmentation, not model weights.
    tokenizer_text = out / "tokenizer_corpus.tmp"
    with tokenizer_text.open("w", encoding="utf-8") as stream:
        for index, record in indexed:
            for pre, answer in examples_for_record(record, index, supported):
                stream.write(pre + answer + "\n")
    tokenizer = JokeTokenizer.train_from_file(
        tokenizer_text, vocab_size=args.bpe_vocab_size,
        extra_special_tokens=train_ids + val_ids,
    )
    tokenizer.save(out / "tokenizer.json")
    tokenizer_text.unlink()
    train_samples, skipped_train = prepare_examples(train_rows, tokenizer, args.block_size, supported)
    val_samples, skipped_val = prepare_examples(validation_rows, tokenizer, args.block_size, supported)
    if not train_samples or not val_samples:
        raise ValueError("No usable training or validation examples; increase --block_size")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head,
        n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    data_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    info = {
        "model": asdict(config),
        "training": vars(args),
        "dataset": {"source_sha256": data_hash, "records": len(records),
                    "training_records": len(train_rows), "validation_records": len(validation_rows),
                    "train_examples": len(train_samples), "validation_examples": len(val_samples),
                    "skipped_too_long": skipped_train + skipped_val},
        "routing": {"topic_patterns": {t: topics_config["topic_keyword_patterns"][t]
                                       for t in sorted(supported)},
                    "topic_counts": {t: counts[t] for t in sorted(supported)}},
        "train_joke_id_tokens": train_ids,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_note": "Train loss selects checkpoints to favor coherent memorized jokes; held-out loss measures generalization separately.",
    }
    (out / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    best_train = float("inf")
    log(f"Model: {model_summary(model, args.device)}", step=0, total=args.max_iters)
    log(f"Training examples={len(train_samples):,}; validation examples={len(val_samples):,}; "
        f"skipped too long={skipped_train + skipped_val}; "
        f"batch={args.batch_size}; learning rate={args.learning_rate:g}; "
        f"joke-ID loss weight={args.id_loss_weight:g}", step=0, total=args.max_iters)
    log("Training a generative model; inference never loads joke records.",
        step=0, total=args.max_iters)
    started = time.perf_counter()
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0 or step == args.max_iters:
            train_loss = estimate_loss(model, train_samples, args.eval_iters, args.batch_size, args.device, args.id_loss_weight)
            val_loss = estimate_loss(model, val_samples, args.eval_iters, args.batch_size, args.device, args.id_loss_weight)
            elapsed = time.perf_counter() - started
            speed = step / elapsed if step else 0.0
            eta = (args.max_iters - step) / speed if speed > 0 else None
            log(f"completion train loss={train_loss:.3f}; validation loss={val_loss:.3f}; "
                f"elapsed={format_duration(elapsed)}; "
                f"speed={speed:.2f} steps/s; ETA={format_duration(eta)}",
                step=step, total=args.max_iters)
            if train_loss < best_train:
                best_train = train_loss
                torch.save(model.state_dict(), out / "model.pt")
                info["checkpoint"] = {"step": step, "train_loss": train_loss, "val_loss": val_loss}
                (out / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
                log(f"Saved improved checkpoint: train loss={train_loss:.3f}; "
                    f"validation loss={val_loss:.3f}", step=step, total=args.max_iters)
        if step == args.max_iters:
            break
        x, y = batch_from_examples(train_samples, args.batch_size, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = completion_loss(model, x, y, args.id_loss_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - started
    checkpoint = info["checkpoint"]
    log(f"Training finished in {format_duration(elapsed)}. "
        f"Best checkpoint: step={checkpoint['step']}/{args.max_iters}; "
        f"train loss={checkpoint['train_loss']:.3f}; "
        f"validation loss={checkpoint['val_loss']:.3f}",
        step=args.max_iters, total=args.max_iters)
    log(f"Saved model configuration: {model_summary(model, args.device)}",
        step=args.max_iters, total=args.max_iters)
    log(f"Saved checkpoint to {out.resolve()}: "
        f"model.pt={((out / 'model.pt').stat().st_size / 1024**2):.2f} MiB; "
        "tokenizer.json; config.json. No dataset required for inference.",
        step=args.max_iters, total=args.max_iters)


if __name__ == "__main__":
    main()
