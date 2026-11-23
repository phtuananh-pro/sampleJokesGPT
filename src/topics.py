"""Map a user's joke request to the exact prompt format used during training.

Topic classification is intentionally simple and transparent: regular
expressions inspect the request, not a joke database. The same canonical
prompts must be used in training and inference for each language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .i18n import GENERAL_PROMPTS, STYLE_PROMPTS, TOPIC_PROMPT_TEMPLATES


@dataclass(frozen=True)
class JokeRequest:
    """Resolved generation prompt and optional topic/keyword diagnostics."""
    prompt: str
    topic: str | None
    keyword: str | None


# Each language has its own preposition for the requested joke subject.
SUBJECT_PREFIXES = {
    "en": r"\b(?:about|on)\s+(.+)",
    "fr": r"\b(?:sur|concernant|à\s+propos\s+de|au\s+sujet\s+de)\s+(.+)",
    "vi": r"\b(?:về|liên\s+quan\s+đến)\s+(.+)",
}

# Styles are independent of subjects: a user can ask for a pun or a joke
# about food. Prefer a subject when the request contains both.
STYLE_REQUEST_PATTERNS = {
    "en": (
        ("pun", r"\b(?:pun|puns|punny)\b"),
        ("dad_joke", r"\b(?:dad\s+jokes?|groaner)\b"),
        ("one_liner", r"\b(?:one[- ]liners?|short\s+joke)\b"),
        ("wordplay", r"\b(?:wordplay|play\s+on\s+words)\b"),
    ),
    "fr": (
        ("pun", r"\b(?:calembours?|jeux?\s+de\s+mots)\b"),
        ("dad_joke", r"\b(?:blagues?\s+de\s+papa)\b"),
        ("one_liner", r"\b(?:blagues?\s+courtes?|blagues?\s+rapides?)\b"),
        ("wordplay", r"\b(?:jeux?\s+de\s+mots)\b"),
    ),
    "vi": (
        ("pun", r"\b(?:chơi\s+chữ|câu\s+chơi\s+chữ)\b"),
        ("dad_joke", r"\b(?:chuyện\s+cười\s+kiểu\s+bố|truyện\s+cười\s+kiểu\s+bố)\b"),
        ("one_liner", r"\b(?:chuyện\s+cười\s+ngắn|truyện\s+cười\s+ngắn)\b"),
        ("wordplay", r"\b(?:nói\s+lái|chơi\s+chữ)\b"),
    ),
}


def topic_prompt(topic: str, language: str = "en",
                 topic_labels: dict[str, str] | None = None) -> str:
    """Create a topic request identical to the one seen by the model.

    Stable topic IDs remain English keys in JSONL; only their human-readable
    prompt label is translated. An older English dataset needs no label table.
    """
    label = (topic_labels or {}).get(topic, topic.replace("_", " and "))
    return TOPIC_PROMPT_TEMPLATES[language].format(topic=label)


def topic_matches(text: str, pattern: str) -> bool:
    """Check whether a decoded answer contains a subject keyword.

    This is only lexical evidence of relevance, not proof of humor quality.
    """
    return re.search(r"\b(?:" + pattern + r")\b", text, re.IGNORECASE) is not None


def route_request(text: str, topic_patterns: dict[str, str],
                  language: str = "en",
                  topic_labels: dict[str, str] | None = None) -> JokeRequest:
    """Route free-form input into a training-compatible language-specific prompt.

    When several topics match, prefer the longest matching phrase, then the
    earliest. This heuristic works without an LLM or retrieval system but can
    misinterpret polysemous words or miss unlisted synonyms.
    """
    if language not in GENERAL_PROMPTS:
        raise ValueError(f"Unsupported language: {language}")

    # A phrase following about/sur/về usually expresses user intent.
    lower = text.lower()
    match = re.search(SUBJECT_PREFIXES[language], lower, re.IGNORECASE)
    focus = match.group(1) if match else lower

    # With no explicit subject ("about", "sur", "về"), treat phrases such as
    # "jeu de mots" / "chơi chữ" as a HUMOR STYLE, not a language topic.
    # If a subject is explicit ("pun about cats"), subject routing wins.
    if not match:
        for style, pattern in STYLE_REQUEST_PATTERNS[language]:
            if re.search(pattern, lower, re.IGNORECASE):
                return JokeRequest(STYLE_PROMPTS[language][style], None, None)

    def candidates_for(fragment: str) -> list[tuple[int, int, str, str]]:
        matches = []
        for topic, pattern in topic_patterns.items():
            hit = re.search(r"\b(?:" + pattern + r")\b", fragment, re.IGNORECASE)
            if hit:
                matches.append((len(hit.group()), -hit.start(), topic, hit.group().lower()))
        return matches

    candidates = candidates_for(focus)
    if not candidates and focus != lower:
        candidates = candidates_for(lower)
    if candidates:
        # The topic ID is a consistent tie-breaker across repeated runs.
        _, _, topic, keyword = max(candidates)
        return JokeRequest(topic_prompt(topic, language, topic_labels), topic, keyword)

    for style, pattern in STYLE_REQUEST_PATTERNS[language]:
        if re.search(pattern, lower, re.IGNORECASE):
            return JokeRequest(STYLE_PROMPTS[language][style], None, None)
    return JokeRequest(GENERAL_PROMPTS[language], None, None)
