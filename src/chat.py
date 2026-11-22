"""A joke chatbot that generates from neural weights, without retrieval.

The decoder first predicts an atomic joke identifier, then generates its text
one token at a time until <|eos|>. The identifier is learned like any other
vocabulary token; it is NEVER used to fetch a joke from a dataset or index.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import GPT, GPTConfig
from .tokenizer import JokeTokenizer
from .topics import JokeRequest, route_request, topic_matches


@dataclass
class Candidate:
    text: str
    finished: bool
    topic_match: bool
    keyword_match: bool


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def load_model(path: Path, device: torch.device):
    """Load only weights, BPE vocabulary, and routing metadata (no jokes)."""
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    tokenizer = JokeTokenizer.load(path / "tokenizer.json")
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    model.load_state_dict(torch.load(path / "model.pt", map_location=device, weights_only=True))
    model.eval()
    ids = [tokenizer.token_id(token) for token in cfg["train_joke_id_tokens"]]
    if not ids:
        raise ValueError("Checkpoint contains no learned joke identities")
    return model, tokenizer, cfg, ids


@torch.no_grad()
def next_token_logits(model: GPT, history: torch.Tensor) -> torch.Tensor:
    """Predict the next-token distribution from a causal context window."""
    logits, _ = model(history[:, -model.config.block_size:])
    return logits[0, -1, :]


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Sample one token or use greedy decoding when temperature is zero."""
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
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
    prefix = f"<|bos|><|user|>{prompt}<|assistant|>"
    history = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)
    choices = next_token_logits(model, history)[allowed_joke_ids]
    choice = sample_from_logits(choices, id_temperature, min(top_k, len(allowed_joke_ids)))
    identity_id = allowed_joke_ids[choice]
    history = torch.cat((history, torch.tensor([[identity_id]], device=device)), dim=1)

    eos_id = tokenizer.token_id("<|eos|>")
    invalid_tokens = {tokenizer.token_id(tok) for tok in
                      ("<|bos|>", "<|user|>", "<|assistant|>", "<|pad|>", "<|unk|>")}
    invalid_tokens.update(allowed_joke_ids)
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
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 5:
        return False
    tetragrams = list(zip(words, words[1:], words[2:], words[3:]))
    if len(tetragrams) > len(set(tetragrams)) + 1:
        return False
    return True


def keyword_matches(text: str, keyword: str | None) -> bool:
    if keyword is None:
        return False
    stem = keyword.lower().rstrip("s") if len(keyword) > 4 else keyword.lower()
    return re.search(r"\b" + re.escape(stem) + r"\w*\b", text, re.IGNORECASE) is not None


def quality_score(candidate: Candidate, requested: JokeRequest) -> float:
    """Favor literal keyword evidence, then topical evidence and readability."""
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
    topic_patterns = cfg["routing"]["topic_patterns"]
    intent = route_request(request, topic_patterns)
    candidates: list[Candidate] = []
    for attempt in range(attempts):
        # A little randomness in later attempts can repair rigid, broken loops.
        temp = body_temperature if attempt < attempts // 2 else max(body_temperature, 0.25)
        text, finished = generate_candidate(
            model, tokenizer, intent.prompt, joke_ids, device,
            id_temperature, temp, max_new_tokens, top_k,
        )
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
        if not intent.topic:
            return item.text, "general"
        if item.keyword_match and item.topic_match:
            return item.text, "topic"
    if not candidates:
        return "I couldn't generate a complete joke. Try again.", "incomplete"
    if intent.topic:
        matches = [item for item in candidates if item.topic_match]
        if matches:
            selected = max(matches, key=lambda item: quality_score(item, intent))
            return selected.text, "topic"
        # Unrelated joke is an explicit, visible fallback, not a silent success.
        selected = max(candidates, key=lambda item: quality_score(item, intent))
        return "I couldn't find a good match, but here's another joke: " + selected.text, "fallback"
    selected = max(candidates, key=lambda item: quality_score(item, intent))
    return selected.text, "general"


def parse_args():
    p = argparse.ArgumentParser(description="Chat with the joke GPT (no retrieval)")
    p.add_argument("--model_dir", default="models/jokes_v2")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=240)
    p.add_argument("--id_temperature", type=float, default=0.75)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Answer-body temperature; 0 favors a single learned full joke")
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    model, tokenizer, cfg, ids = load_model(Path(args.model_dir), device)
    print(f"JokeGPT v2 ready on {device}. Types supported: "
          f"{len(cfg['routing']['topic_patterns'])} topics. Type exit to quit.")
    you_label = "\033[96mYou:\033[0m"
    bot_label = "\033[93mJoke bot:\033[0m"
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
        answer, status = generate_best(
            model, tokenizer, cfg, ids, device, query, args.attempts,
            args.max_new_tokens, args.id_temperature, args.temperature, args.top_k,
        )
        print(f"\n{bot_label} {answer}")


if __name__ == "__main__":
    main()
