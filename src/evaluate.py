"""Measure how often neural generation produces complete and on-topic jokes.

This evaluates the *model's outputs*, not dataset retrieval. Complete means
that generation reached EOS and passed lightweight format checks; on-topic
means a lexical topic match. Neither metric measures funniness reliably.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chat import choose_device, generate_best, load_model

PROMPTS = [
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/jokes_v2")
    p.add_argument("--out", default="models/jokes_v2/evaluation.json")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--attempts", type=int, default=10)
    args = p.parse_args()
    device = choose_device(args.device)
    model, tokenizer, cfg, ids = load_model(Path(args.model_dir), device)
    results = []
    for prompt in PROMPTS:
        answer, status = generate_best(model, tokenizer, cfg, ids, device,
                                       prompt, args.attempts, 240, 0.75, 0.0, 20)
        results.append({"prompt": prompt, "status": status, "answer": answer})
        print(f"{status:10} {prompt}\n  {answer}\n", flush=True)
    counts = {kind: sum(r["status"] == kind for r in results)
              for kind in ("general", "topic", "fallback", "incomplete")}
    payload = {"count": len(results), "status_counts": counts, "results": results,
               "limits": "EOS and lexical topicality proxies are not a human joke-quality review."}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Metrics:", counts)


if __name__ == "__main__":
    main()
