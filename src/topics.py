"""Shared topic prompts and weak keyword labels for joke examples.

The same patterns label jokes during dataset preparation and route user
requests at inference time, keeping both sides on matching topic names.
"""

from __future__ import annotations

import re


TOPIC_PROMPTS = {
    "programming": "Tell me a joke about programming.",
    "teacher": "Tell me a joke about teachers.",
    "animal": "Tell me a joke about animals.",
}

TOPIC_PATTERNS = {
    "programming": re.compile(
        r"\b(programmer|programmers|programming|developer|developers|"
        r"coding|code|python|java|javascript|software|computer|computers|"
        r"bug|bugs|keyboard|server|website)\b", re.IGNORECASE,
    ),
    "teacher": re.compile(
        r"\b(teacher|teachers|school|schools|classroom|classrooms|student|"
        r"students|professor|professors|principal|homework|report card)\b",
        re.IGNORECASE,
    ),
    "animal": re.compile(
        r"\b(animal|animals|cat|cats|kitten|kittens|dog|dogs|puppy|puppies|"
        r"bird|birds|fish|horse|horses|cow|cows|pig|pigs|duck|ducks|"
        r"chicken|chickens|bear|bears|rabbit|rabbits|mouse|mice|frog|frogs|"
        r"snake|snakes|lion|lions|elephant|elephants|monkey|monkeys)\b",
        re.IGNORECASE,
    ),
}


def detect_topics(text: str) -> list[str]:
    """Return every topic whose whole-word keywords occur in the text."""
    return [
        topic for topic, pattern in TOPIC_PATTERNS.items()
        if pattern.search(text)
    ]
