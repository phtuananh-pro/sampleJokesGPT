"""Evaluate the trained JokeGPT in the selected language.

Only model predictions are evaluated. Lexical topic status and EOS stopping
help diagnose decoding, but are not a substitute for judging whether a joke
sounds natural and funny in French, Vietnamese, or English.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chat import choose_device, generate_best, load_model
from .i18n import LANGUAGES, GENERAL_PROMPTS, STYLE_PROMPTS, chat_model_dir
from .topics import topic_prompt


# Existing English benchmark requests remain unchanged for comparability.
ENGLISH_PROMPTS = [
    "Tell me a joke.", "Tell me a pun.", "Tell me a dad joke.",
    "Tell me a joke about programmers.", "Joke about teachers.",
    "Tell me a joke about dogs.", "Tell me a joke about cats.",
    "Tell me a joke about books.", "Tell me a joke about food.",
    "Tell me a joke about music.", "Tell me a joke about math.",
    "Tell me a joke about birds.", "Tell me a joke about doctors.",
    "Tell me a joke about the weather.", "Tell me a joke about space.",
    "Tell me a joke about sports.", "Tell me a joke about travel.",
    "Tell me a joke about cooking.", "Tell me a joke about the ocean.",
    "Tell me a joke about computers.",
]


def prompts_for_language(language: str, cfg: dict) -> list[str]:
    """Construct locale-correct prompts from the checkpoint's trained topics.

    Unlike a hardcoded English test set, this avoids evaluating subjects that
    the French/Vietnamese model never learned.
    """
    if language == "en":
        return ENGLISH_PROMPTS
    prompts = [GENERAL_PROMPTS[language], *STYLE_PROMPTS[language].values()]
    labels = cfg["routing"].get("topic_labels", {})
    for topic in cfg["routing"]["topic_patterns"]:
        prompts.append(topic_prompt(topic, language, labels))
    return prompts[:20]


def main() -> None:
    """Run deterministic diagnostic requests and save human-readable answers."""
    p = argparse.ArgumentParser(description="Evaluate JokeGPT model generations")
    p.add_argument("-l", "--language", default="en", choices=LANGUAGES)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--attempts", type=int, default=10)
    args = p.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else chat_model_dir(args.language)
    cfg_path = model_dir / "config.json"
    cfg_meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    checkpoint_lang = cfg_meta.get(
        "language", cfg_meta.get("training", {}).get("language", "en"))
    if checkpoint_lang != args.language:
        raise ValueError(
            f"Checkpoint language={checkpoint_lang}, requested={args.language}"
        )

    device = choose_device(args.device)
    model, tokenizer, cfg, ids = load_model(model_dir, device)
    results = []
    for prompt in prompts_for_language(args.language, cfg):
        answer, status = generate_best(
            model, tokenizer, cfg, ids, device, prompt, args.attempts,
            240, 0.75, 0.0, 20,
        )
        results.append({"prompt": prompt, "status": status, "answer": answer})
        print(f"{status:10} {prompt}\n  {answer}\n", flush=True)

    counts = {
        key: sum(r["status"] == key for r in results)
        for key in ("general", "topic", "fallback", "incomplete")
    }
    payload = {
        "language": args.language, "count": len(results),
        "status_counts": counts, "results": results,
        "limits": "EOS and lexical topicality are not human humor-quality assessments.",
    }
    destination = Path(args.out) if args.out else model_dir / "evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics ({args.language}): {counts}; saved {destination}")


if __name__ == "__main__":
    main()
