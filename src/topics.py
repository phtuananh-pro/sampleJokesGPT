"""Shared topic vocabulary and weak, transparent corpus labels.

Labels are keyword heuristics, not semantic understanding. Keep the patterns
specific enough that 'python' as an animal, or 'bug' as an insect, does not
automatically become a programming joke. Topic classification remains noisy.
"""

from __future__ import annotations

import re


GENERAL_PROMPTS = ("Tell me a joke.", "Make me laugh.", "Tell me another joke.")

TOPIC_VARIANTS = {
    "programming": (
        "Tell me a joke about programming.",
        "Tell me a programming joke.",
        "Make me laugh about programmers.",
        "Tell me a joke about coding.",
    ),
    "teacher": (
        "Tell me a joke about teachers.",
        "Tell me a teacher joke.",
        "Make me laugh about school.",
    ),
    "animal": (
        "Tell me a joke about animals.",
        "Tell me an animal joke.",
        "Make me laugh about animals.",
    ),
    "children": (
        "Tell me a joke about children.",
        "Tell me a joke about kids.",
        "Make me laugh about children.",
    ),
    "work": (
        "Tell me a joke about work.",
        "Tell me an office joke.",
        "Make me laugh about my job.",
    ),
}

TOPIC_PROMPTS = {topic: prompts[0] for topic, prompts in TOPIC_VARIANTS.items()}

TOPIC_PATTERNS = {
    "programming": re.compile(
        r"\b(programmer|programmers|programming|developer|developers|"
        r"coding|coded|computer|computers|software|javascript|keyboard|"
        r"server|servers|website|websites|algorithm|algorithms|"
        r"compile|compiler|debugging|git|github|database|databases)\b",
        re.IGNORECASE,
    ),
    "teacher": re.compile(
        r"\b(teacher|teachers|school|schools|classroom|classrooms|"
        r"student|students|professor|professors|principal|homework|"
        r"report card|exam|exams)\b",
        re.IGNORECASE,
    ),
    "animal": re.compile(
        r"\b(animal|animals|cat|cats|kitten|kittens|dog|dogs|"
        r"puppy|puppies|bird|birds|fish|horse|horses|cow|cows|"
        r"pig|pigs|duck|ducks|chicken|chickens|bear|bears|"
        r"rabbit|rabbits|mouse|mice|frog|frogs|snake|snakes|"
        r"lion|lions|elephant|elephants|monkey|monkeys|"
        r"tiger|tigers|zebra|zebras|giraffe|giraffes)\b",
        re.IGNORECASE,
    ),
    "children": re.compile(
        r"\b(child|children|kid|kids|baby|babies|toddler|toddlers|"
        r"parent|parents|son|daughter|sons|daughters)\b",
        re.IGNORECASE,
    ),
    "work": re.compile(
        r"\b(work|working|workplace|office|offices|boss|bosses|"
        r"employee|employees|coworker|coworkers|colleague|colleagues|"
        r"manager|managers|meeting|meetings|salary|job|jobs)\b",
        re.IGNORECASE,
    ),
}


def detect_topics(text: str) -> list[str]:
    """Find all explicitly mentioned categories, ordered by dictionary priority."""
    return [topic for topic, pattern in TOPIC_PATTERNS.items() if pattern.search(text)]


def detect_requested_topic(text: str) -> str | None:
    """Find one topic in the request; prefer later, explicitly topical phrases.

    Requests containing multiple categories remain ambiguous and choose the
    first detected category. The returned topic is only a coarse routing hint.
    """

    matches = detect_topics(text)
    return matches[0] if matches else None
