"""Interactive joke chatbot with candidate generation and simple ranking.

The model generates continuations from a serialized user/assistant prompt.
This script handles the surrounding conversation, device selection, stopping
at chat markers, and a lightweight heuristic that chooses among samples.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import JokeTokenizer
from .topics import TOPIC_PROMPTS, detect_requested_topic, detect_topics


def choose_device(requested: str) -> torch.device:
    """Choose CUDA automatically when available, otherwise CPU; validate CUDA."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(requested)


def detect_intent(user_text: str) -> str:
    """Map a request to the general or topic prompt used by the cleaner."""
    return detect_requested_topic(user_text) or "general"


def build_prompt(user_text: str) -> str:
    """Build the training-compatible chat prefix for the detected joke type."""
    intent = detect_intent(user_text)
    request = TOPIC_PROMPTS.get(intent, "Tell me a joke.")
    return f"<|bos|>\n<|user|>{request}\n<|assistant|>\n"


def extract_answer(full_text: str) -> str:
    """Keep only generated assistant text before the next chat boundary."""
    text = full_text.split("<|assistant|>", 1)[-1]
    for marker in ("<|eos|>", "<|bos|>", "<|user|>", "<|assistant|>"):
        text = text.split(marker, 1)[0]
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def has_repetition(text: str) -> bool:
    """Detect repeated word n-grams that often indicate a broken generation."""
    words = text.lower().split()
    if len(words) < 10:
        return False
    for size in (2, 3, 4):
        chunks = [tuple(words[i : i + size]) for i in range(len(words) - size + 1)]
        if len(chunks) != len(set(chunks)):
            return True
    return False


def is_good_joke(text: str) -> bool:
    """Filter empty, overlong, repetitive, or unclosed-marker candidates."""
    return (
        20 <= len(text) <= 450
        and "<|" not in text
        and "|>" not in text
        and not has_repetition(text)
        and len(set(text)) >= 8
    )


def score_joke(text: str) -> float:
    """Prefer reasonably sized and structurally readable candidate answers.

    This is NOT a judge of humor, grammatical coherence, or punchline quality.
    """
    score = 0.0
    length = len(text)
    if 50 <= length <= 220:
        score += 3.0
    if "?" in text:
        score += 2.0
    if "." in text or "!" in text:
        score += 1.0
    if "\n" in text:
        score += 0.5
    if has_repetition(text):
        score -= 5.0
    if length > 300:
        score -= 2.0
    return score


@torch.no_grad()
def generate_one(
    model: GPT, tokenizer: JokeTokenizer, prompt: str, device: torch.device,
    max_new_tokens: int, temperature: float, top_k: int,
) -> str:
    """Sample one response and remove the serialized chat prefix and markers."""
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    eos_id = tokenizer.tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <|eos|>.")
    generated = model.generate(
        x, max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k, eos_token_id=eos_id,
    )
    full_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
    return extract_answer(full_text)


def generate_best(
    model: GPT, tokenizer: JokeTokenizer, prompt: str, device: torch.device,
    attempts: int, max_new_tokens: int, temperature: float, top_k: int,
    topic: str | None = None, recent: set[str] | None = None,
) -> str:
    """Generate multiple completions; optionally require a visible topic cue.

    No training-joke retrieval is performed. Word-based relevance checks can
    reject subtle jokes and still accept irrelevant ones; they are only a
    transparent first-pass safeguard against obviously off-topic responses.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")
    candidates = []
    for _ in range(attempts):
        joke = generate_one(model, tokenizer, prompt, device, max_new_tokens, temperature, top_k)
        if not is_good_joke(joke):
            continue
        if recent and joke.casefold() in recent:
            continue
        if topic is not None and topic not in detect_topics(joke):
            continue
        candidates.append(joke)
    if not candidates:
        if topic is not None:
            return f"I couldn't generate a relevant joke about {topic}. Try again."
        return "I could not generate a complete joke. Try again."
    return max(candidates, key=score_joke)


def load_model(model_dir: Path, device: torch.device) -> tuple[GPT, JokeTokenizer]:
    """Load a saved architecture, weights, and matching tokenizer onto device."""
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.pt"
    tokenizer_path = model_dir / "tokenizer.json"
    for path in (config_path, model_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    model = GPT(GPTConfig(**config_data["model"])).to(device)
    tokenizer = JokeTokenizer.load(tokenizer_path)
    # Checkpoints contain tensor state only; weights_only avoids unpickling
    # arbitrary Python objects when loading the user-selected model folder.
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    checkpoint = config_data.get("checkpoint", {})
    print(
        "Loaded model: "
        f"{sum(p.numel() for p in model.parameters()):,} parameters | "
        f"{model.config.n_layer} layers | "
        f"{model.config.n_head} heads | "
        f"embedding {model.config.n_embd} | "
        f"context {model.config.block_size} | "
        f"vocab {model.config.vocab_size} | "
        f"dropout {model.config.dropout} | device {device}"
    )
    if checkpoint:
        print(
            f"Checkpoint: step {checkpoint.get('best_step', '?')}, "
            f"validation loss {checkpoint.get('best_val_loss', '?')}"
        )
    return model, tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    """Define checkpoint, device, sampling, and candidate-count options."""
    parser = argparse.ArgumentParser(description="Chat with a trained joke GPT model.")
    parser.add_argument("--model_dir", default="models/jokes_bpe")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="auto selects CUDA when available, otherwise CPU",
    )
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=40)
    return parser


def main() -> None:
    """Load the checkpoint and serve joke requests until the user exits."""
    args = build_arg_parser().parse_args()
    device = choose_device(args.device)
    model, tokenizer = load_model(Path(args.model_dir), device)
    print(f"Joke GPT ready on {device}. Type 'exit' to quit.")

    # ANSI colors apply to the labels only; input and generated text use the
    # terminal's normal foreground color.
    you_label = "\033[96mYou:\033[0m"
    bot_label = "\033[93mJoke bot:\033[0m"
    recent_answers: set[str] = set()

    while True:
        user_text = input(f"\n{you_label} ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        if not user_text:
            print("Please enter a request, such as 'tell me a joke about teachers'.")
            continue
        topic = detect_intent(user_text)
        answer = generate_best(
            model, tokenizer, build_prompt(user_text), device,
            args.attempts, args.max_new_tokens, args.temperature, args.top_k,
            topic=topic if topic != "general" else None,
            recent=recent_answers,
        )
        if not answer.startswith("I couldn't") and not answer.startswith("I could not"):
            recent_answers.add(answer.casefold())
            if len(recent_answers) > 50:
                recent_answers.clear()
        print(f"\n{bot_label} {answer}")


if __name__ == "__main__":
    main()
