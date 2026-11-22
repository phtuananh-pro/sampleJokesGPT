import argparse
import html
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .topics import GENERAL_PROMPTS, TOPIC_VARIANTS, detect_topics


def normalize_text(text: str) -> str:
    """Normalize Unicode, HTML entities, punctuation, and repeated whitespace."""
    text = text.replace("\ufeff", "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({
        "“": '"', "”": '"', "„": '"',
        "‘": "'", "’": "'", "–": "-", "—": "-",
    }))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_possible_id(text: str) -> str:
    """Remove a leading numeric record ID separated by common delimiters."""
    # Handles lines like:
    # 123,Why did...
    # 123\tWhy did...
    # 123;Why did...
    text = re.sub(r"^\s*\d+\s*[,;\t|]\s*", "", text)
    return text.strip()


def is_valid_joke(text: str, min_chars: int, max_chars: int) -> bool:
    """Reject empty, unusually sized, URL-heavy, or non-text records."""
    if len(text) < min_chars:
        return False
    if len(text) > max_chars:
        return False
    if text.count("http") > 0:
        return False
    if len(set(text)) < 8:
        return False
    if re.fullmatch(r"[\W\d_]+", text):
        return False
    return True


def to_training_sample(joke: str, prompt: str) -> str:
    """Serialize one user request and joke with explicit chat boundary tokens."""
    return (
        "<|bos|>\n"
        f"<|user|>{prompt}\n"
        "<|assistant|>\n"
        f"{joke}\n"
        "<|eos|>\n"
    )


def prompts_for_joke(joke: str) -> list[str]:
    """Augment each complete joke with varied requests matching its detected topic.

    Topic recognition is approximate; manually inspect examples for label noise.
    """
    prompts = list(GENERAL_PROMPTS)
    for topic in detect_topics(joke):
        prompts.extend(TOPIC_VARIANTS[topic])
    return prompts


def main():
    """Read raw jokes, deduplicate and shuffle them, then write training text."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.txt")
    parser.add_argument("--output", default="data_clean.txt")
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--max_chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    seen = set()
    jokes = []

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            joke = normalize_text(line)
            joke = remove_possible_id(joke)
            joke = normalize_text(joke)

            if not is_valid_joke(joke, args.min_chars, args.max_chars):
                continue

            key = joke.lower()
            key = re.sub(r"\s+", " ", key)

            if key in seen:
                continue

            seen.add(key)
            jokes.append(joke)

    random.shuffle(jokes)
    topic_counts = Counter()

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for joke in jokes:
            prompts = prompts_for_joke(joke)
            for prompt in prompts:
                f.write(to_training_sample(joke, prompt))
                f.write("\n")
                if prompt not in GENERAL_PROMPTS:
                    topic_counts[prompt] += 1

    print(f"Input lines cleaned: {len(jokes)}")
    print(f"Training examples written: {len(jokes) + sum(topic_counts.values())}")
    print("Topic-labeled examples (keyword-based):")
    for topic, prompts in TOPIC_VARIANTS.items():
        count = sum(topic_counts[prompt] for prompt in prompts)
        print(f"  {topic}: {count:,} labeled request/joke examples")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
"""Clean one-joke-per-line text into the chatbot's supervised dialogue form.

The cleaner normalizes dataset text and creates user/assistant markers. This
is data preparation for the self-supervised next-token objective: the model
still learns every target from the following token in this serialized stream.
"""

