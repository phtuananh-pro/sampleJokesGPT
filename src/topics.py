"""Convert natural-language joke requests into the training topic vocabulary.

Topic routing reads a *taxonomy*, never a collection of jokes. No retrieval is
performed. The same topic name is used in training and inference so the GPT
can learn its association with joke completions (FMg Part IV: conditioning).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class JokeRequest:
    """A canonical training prompt plus the detected domain, if supported."""
    prompt: str
    topic: str | None
    keyword: str | None


def topic_prompt(topic: str) -> str:
    """Build exactly the same prompt in training and interactive inference."""
    return f"Tell me a joke about {topic.replace('_', ' and ')}."


def topic_matches(text: str, pattern: str) -> bool:
    """Check a label's lexical pattern against text with word boundaries."""
    return re.search(r"\b(?:" + pattern + r")\b", text, re.IGNORECASE) is not None


def route_request(text: str, topic_patterns: dict[str, str]) -> JokeRequest:
    """Pick the strongest known topic named in the user's request.

    This lightweight lexical classifier is intentionally transparent. It can
    confuse polysemous words (e.g. a software bug vs. an insect); a model-based
    intent classifier would be a separate, more advanced system.
    """
    lower = text.lower()
    # Content after "about" or "on" is usually more informative than "joke".
    match = re.search(r"\b(?:about|on)\s+(.+)", lower)
    focus = match.group(1) if match else lower
    candidates = []
    for topic, pattern in topic_patterns.items():
        found = re.search(r"\b(?:" + pattern + r")\b", focus, re.IGNORECASE)
        if found:
            candidates.append((len(found.group()), -found.start(), topic))
    if not candidates and focus != lower:
        for topic, pattern in topic_patterns.items():
            found = re.search(r"\b(?:" + pattern + r")\b", lower, re.IGNORECASE)
            if found:
                candidates.append((len(found.group()), -found.start(), topic))
    if candidates:
        candidates.sort(reverse=True)
        _, _, topic = candidates[0]
        # Record the actual matched phrase, useful for ranking generated jokes.
        matched = re.search(r"\b(?:" + topic_patterns[topic] + r")\b", focus, re.IGNORECASE)
        keyword = matched.group().lower() if matched else None
        return JokeRequest(topic_prompt(topic), topic, keyword)
    if re.search(r"\b(?:pun|puns|punny)\b", lower):
        return JokeRequest("Tell me a pun.", None, None)
    if re.search(r"\b(?:dad joke|dad jokes|groaner)\b", lower):
        return JokeRequest("Tell me a dad joke.", None, None)
    if re.search(r"\b(?:one[- ]liner|short joke)\b", lower):
        return JokeRequest("Tell me a one-liner.", None, None)
    return JokeRequest("Tell me a joke.", None, None)
