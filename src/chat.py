"""A joke chatbot that generates from neural weights, without retrieval.

The decoder first predicts an atomic joke identifier, then generates its text
one token at a time until <|eos|>. The identifier is learned like any other
vocabulary token; it is NEVER used to fetch a joke from a dataset or index.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT, GPTConfig
from .tokenizer import JokeTokenizer
from .topics import JokeRequest, route_request, topic_matches
from .i18n import (
    LANGUAGES, FALLBACK_PREFIXES, INCOMPLETE_MESSAGES, SPEAKER_LABELS,
    chat_model_dir,
)


def log(message: str) -> None:
    """Timestamp inference diagnostics without changing the colored chat labels."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def describe_loaded_model(model: GPT, tokenizer: JokeTokenizer, cfg: dict,
                          joke_ids: list[int], device: torch.device, model_dir: Path) -> None:
    """Print architecture, checkpoint quality, and generation capabilities."""
    arch = model.config
    params = sum(p.numel() for p in model.parameters())
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    checkpoint = cfg.get("checkpoint", {})
    log(f"Loaded model from {model_dir.resolve()} on {device} ({device_name})")
    log(f"Architecture: {params:,} parameters ({params / 1e6:.2f}M); "
        f"layers={arch.n_layer}; heads={arch.n_head}; embedding={arch.n_embd}; "
        f"context={arch.block_size}; vocab={tokenizer.vocab_size}; dropout={arch.dropout}")
    log(f"Capabilities: {len(joke_ids):,} trained joke IDs; "
        f"{len(cfg['routing']['topic_patterns'])} supported topics; no retrieval")
    if checkpoint:
        log(f"Checkpoint: step={checkpoint.get('step', 'n/a')}/"
            f"{cfg.get('training', {}).get('max_iters', 'n/a')}; "
            f"train loss={checkpoint.get('train_loss', 'n/a')}; "
            f"validation loss={checkpoint.get('val_loss', 'n/a')}")


@dataclass
class Candidate:
    """One generated answer and its inexpensive completion/topic diagnostics.

    These flags rank *model-generated* text; they do not retrieve source jokes.
    """
    text: str
    finished: bool
    topic_match: bool
    keyword_match: bool


def choose_device(requested: str) -> torch.device:
    """Select CPU/CUDA without silently ignoring an explicit CUDA request."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_model(path: Path, device: torch.device):
    """Load only weights, BPE vocabulary, and routing metadata (no jokes)."""
    # The checkpoint carries architecture and topic rules. The training
    # dataset is deliberately neither opened nor searched at inference.
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tokenizer = JokeTokenizer.load(path / "tokenizer.json")
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    # Deserialize tensor weights only; recreate the exact saved architecture
    # before loading and switch off dropout for reproducible decoding.
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    model.eval()
    # Only IDs seen during training can be sampled. This list contains token
    # IDs, NOT a map from IDs to stored text or another retrieval index.
    ids = [tokenizer.token_id(token) for token in cfg["train_joke_id_tokens"]]
    if not ids:
        raise ValueError("Checkpoint contains no learned joke identities")
    return model, tokenizer, cfg, ids


@torch.no_grad()
def next_token_logits(model: GPT, history: torch.Tensor) -> torch.Tensor:
    """Predict the next-token distribution from a causal context window."""
    # The causal model sees at most the learned positional context window;
    # the last position predicts exactly one next-token distribution.
    logits, _ = model(history[:, -model.config.block_size:])
    return logits[0, -1, :]


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Sample one token or use greedy decoding when temperature is zero."""
    # Temperature=0 deterministically chooses the most probable token.
    # Positive temperature allows controlled stochastic generation.
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    # Restrict candidates to the top-k tokens before softmax, reducing
    # low-probability completions that can corrupt a tiny model's punchline.
    if 0 < top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < cutoff, -float("inf"))
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


@torch.no_grad()
def generate_candidate(model: GPT, tokenizer: JokeTokenizer, prompt: str,
                       allowed_joke_ids: list[int], device: torch.device,
                       id_temperature: float, body_temperature: float,
                       max_new_tokens: int, top_k: int) -> tuple[str, bool]:
    """Sample a learned identity first, then generate its complete wording.

    Constraining the first generated token to a *learned identity token* is a
    vocabulary constraint, not retrieval: neither a joke file nor a table of
    joke texts is loaded. The rest of the answer is produced by the Transformer.
    """
    # Use the *same* serialized prompt grammar seen during training.
    # A mismatched prefix would shift the model out of its learned context.
    prefix = f"<|bos|><|user|>{prompt}<|assistant|>"
    history = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)
    # The first assistant position must be a *learned joke ID*. Constraining
    # vocabulary logits is decoding, not lookup: no joke text is consulted.
    choices = next_token_logits(model, history)[allowed_joke_ids]
    choice = sample_from_logits(choices, id_temperature, min(top_k, len(allowed_joke_ids)))
    identity_id = allowed_joke_ids[choice]
    history = torch.cat((history, torch.tensor([[identity_id]], device=device)), dim=1)

    # EOS is trained as the true boundary of a full joke. A separate token
    # cap exists only to stop pathological or overly long generations.
    eos_id = tokenizer.token_id("<|eos|>")
    # A joke body must not emit another chat role, padding, or joke identity.
    # Such control tokens signal a malformed continuation; reject it.
    invalid_tokens = {tokenizer.token_id(tok) for tok in
                      ("<|bos|>", "<|user|>", "<|assistant|>", "<|pad|>", "<|unk|>")}
    invalid_tokens.update(allowed_joke_ids)
    # Generate autoregressively: each sampled body token becomes part of the
    # next causal context, until EOS or a visible failure condition occurs.
    generated = []
    for _ in range(max_new_tokens):
        logits = next_token_logits(model, history)
        next_id = sample_from_logits(logits, body_temperature, top_k)
        if next_id == eos_id:
            return tokenizer.decode(generated, skip_special_tokens=False).strip(), True
        if next_id in invalid_tokens:
            return tokenizer.decode(generated, skip_special_tokens=False).strip(), False
        generated.append(next_id)
        history = torch.cat((history, torch.tensor([[next_id]], device=device)), dim=1)
    return tokenizer.decode(generated, skip_special_tokens=False).strip(), False


def is_complete_joke(text: str, finished: bool) -> bool:
    """Reject truncated, repetitive, suspiciously short and broken answers."""
    if not finished or not (20 <= len(text) <= 450) or "<|" in text:
        return False
    # EOS is necessary, but it cannot prove a joke is grammatical or funny.
    # Heuristics reject obvious truncation, short fragments and loops.
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return False
    tetragrams = list(zip(words, words[1:], words[2:], words[3:]))
    if len(tetragrams) > len(set(tetragrams)) + 1:
        return False
    return True


def keyword_matches(text: str, keyword: str | None) -> bool:
    """Look for a request keyword in generated text; weak topical evidence."""
    if keyword is None:
        return False
    # Very basic singular/plural handling; not a semantic classifier.
    stem = keyword.lower().rstrip("s") if len(keyword) > 4 else keyword.lower()
    return re.search(r"\b" + re.escape(stem) + r"\w*\b", text, re.IGNORECASE) is not None


def quality_score(candidate: Candidate, requested: JokeRequest) -> float:
    """Favor literal keyword evidence, then topical evidence and readability."""
    # These small readability/keyword preferences are ranking heuristics,
    # not learned reward scores and not a replacement for human review.
    score = 0.0
    if requested.topic:
        score += 10 if candidate.keyword_match else 0
        score += 5 if candidate.topic_match else 0
    if "?" in candidate.text:
        score += 1
    if candidate.text.endswith((".", "!", "?", "\"", "'")):
        score += 1
    if 35 <= len(candidate.text) <= 250:
        score += 1
    return score


def generate_best(model: GPT, tokenizer: JokeTokenizer, cfg: dict,
                  joke_ids: list[int], device: torch.device, request: str,
                  attempts: int, max_new_tokens: int, id_temperature: float,
                  body_temperature: float, top_k: int) -> tuple[str, str]:
    """Try multiple neural generations; prefer complete on-topic answers."""
    # Read ONLY the saved taxonomy from config, never the JSONL dataset.
    topic_patterns = cfg["routing"]["topic_patterns"]
    language = cfg.get("language", "en")
    intent = route_request(
        request, topic_patterns, language,
        cfg["routing"].get("topic_labels", {}),
    )
    candidates: list[Candidate] = []
    # Try a few independent neural completions. This helps with occasional
    # malformed output without altering the model or adding retrieval.
    for attempt in range(attempts):
        # A little randomness in later attempts can repair rigid, broken loops.
        temp = body_temperature if attempt < attempts // 2 else max(body_temperature, 0.25)
        text, finished = generate_candidate(
            model, tokenizer, intent.prompt, joke_ids, device,
            id_temperature, temp, max_new_tokens, top_k,
        )
        # The assistant should display complete jokes, not arbitrary prefixes.
        if not is_complete_joke(text, finished):
            continue
        item = Candidate(
            text=text, finished=True,
            topic_match=(topic_matches(text, topic_patterns[intent.topic]) if intent.topic else False),
            keyword_match=keyword_matches(text, intent.keyword),
        )
        candidates.append(item)
        # Returning a valid joke immediately avoids many expensive decoder
        # steps, especially on CPU (the educational model has no KV cache).
        # Stop early for an acceptable completion to keep CPU inference usable.
        # Topic requests require stronger evidence when it is available.
        if not intent.topic:
            return item.text, "general"
        if item.keyword_match and item.topic_match:
            return item.text, "topic"
    if not candidates:
        return INCOMPLETE_MESSAGES[language], "incomplete"
    if intent.topic:
        matches = [item for item in candidates if item.topic_match]
        if matches:
            selected = max(matches, key=lambda item: quality_score(item, intent))
            return selected.text, "topic"
        # Unrelated joke is an explicit, visible fallback, not a silent success.
        selected = max(candidates, key=lambda item: quality_score(item, intent))
        return FALLBACK_PREFIXES[language] + selected.text, "fallback"
    selected = max(candidates, key=lambda item: quality_score(item, intent))
    return selected.text, "general"


def parse_args():
    """Configure checkpoint loading and decoding without retraining weights."""
    p = argparse.ArgumentParser(description="Chat with the joke GPT (no retrieval)")
    p.add_argument("-l", "--language", choices=LANGUAGES, default="en",
                   help="Chat language: en (default), fr or vi")
    p.add_argument("--model_dir", default=None,
                   help="Explicit checkpoint; otherwise use the selected language")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=240)
    p.add_argument("--id_temperature", type=float, default=0.75)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Answer-body temperature; 0 favors a single learned full joke")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--verbose", action="store_true",
                   help="Show request topic, response status, and latency")
    return p.parse_args()


def main():
    """Load the learned chatbot once, then answer interactive requests."""
    args = parse_args()
    device = choose_device(args.device)
    model_dir = Path(args.model_dir) if args.model_dir else chat_model_dir(args.language)
    if not (model_dir / "model.pt").is_file():
        raise FileNotFoundError(
            f"No trained weights at {model_dir / 'model.pt'} for {args.language}. "
            "Train this language first, or provide a complete --model_dir."
        )
    # Previous English checkpoints did not store a language field. Treat them
    # as English only; never decode French with English-only weights.
    config_file = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_language = config_file.get("language",
                                          config_file.get("training", {}).get("language", "en"))
    if checkpoint_language != args.language:
        raise ValueError(
            f"Checkpoint language={checkpoint_language}, requested language={args.language}. "
            "Train and load a separate model for each language."
        )
    model, tokenizer, cfg, ids = load_model(model_dir, device)
    describe_loaded_model(model, tokenizer, cfg, ids, device, model_dir)
    log(f"Decoding: attempts={args.attempts}; max tokens={args.max_new_tokens}; "
        f"ID temperature={args.id_temperature:g}; body temperature={args.temperature:g}; "
        f"top-k={args.top_k}. Type exit to quit.")
    # ANSI escapes color only the speaker labels, not users' text or jokes.
    user_text, bot_text = SPEAKER_LABELS[args.language]
    you_label = f"\033[96m{user_text}\033[0m"
    bot_label = f"\033[93m{bot_text}\033[0m"
    while True:
        try:
            query = input(f"\n{you_label} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        # A request is routed to a training-compatible prompt; the model
        # returns body tokens. Verbose latency stays outside the joke text.
        started = time.perf_counter()
        answer, status = generate_best(
            model, tokenizer, cfg, ids, device, query, args.attempts,
            args.max_new_tokens, args.id_temperature, args.temperature, args.top_k,
        )
        print(f"\n{bot_label} {answer}")
        if args.verbose:
            intent = route_request(
                query, cfg["routing"]["topic_patterns"], args.language,
                cfg["routing"].get("topic_labels", {}),
            )
            log(f"Response: topic={intent.topic or 'general'}; status={status}; "
                f"latency={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
