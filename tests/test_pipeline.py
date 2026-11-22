"""Minimal tests with the *real* tokenizer installed, plus model-free checks."""
import json
import re
import unittest
from pathlib import Path

from src.topics import route_request, topic_matches, topic_prompt

DATA = Path(__file__).resolve().parents[1] / "data"


class TopicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(s) for s in (DATA / "jokes_clean.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.catalog = json.loads((DATA / "topics.json").read_text(encoding="utf-8"))["topic_keyword_patterns"]

    def test_schema_and_count(self):
        self.assertEqual(len(self.rows), 1440)
        self.assertTrue(all(set(["prompt", "joke", "style", "topics"]) <= set(r) for r in self.rows))

    def test_keyword_to_topic(self):
        self.assertEqual(route_request("Tell me a joke about programmers", self.catalog).topic, "programming")
        self.assertEqual(route_request("Joke about a teacher", self.catalog).topic, "school_education")
        self.assertEqual(route_request("Joke about a dog", self.catalog).topic, "animals_pets")
        self.assertEqual(route_request("Tell me a joke about coffee", self.catalog).topic, "drinks_coffee_tea")

    def test_generic_and_style_prompts(self):
        self.assertEqual(route_request("Tell me a joke", self.catalog).prompt, "Tell me a joke.")
        self.assertEqual(route_request("Give me a pun", self.catalog).prompt, "Tell me a pun.")

    def test_topic_rule_reuse(self):
        self.assertTrue(topic_matches("My cat loves computers", self.catalog["animals_pets"]))
        self.assertFalse(topic_matches("My cat loves computers", self.catalog["school_education"]))
        self.assertEqual(topic_prompt("animals_pets"), "Tell me a joke about animals and pets.")


if __name__ == "__main__":
    unittest.main()
