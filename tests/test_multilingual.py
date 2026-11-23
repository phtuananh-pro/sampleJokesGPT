"""Regression tests for language selection, corpus schema and prompt routing.

These tests are deliberately lightweight: they test configuration invariants
without a long GPU training run or claims about whether the jokes are funny.
"""
import json
import unittest
from pathlib import Path

from src.i18n import (
    LANGUAGES, GENERAL_PROMPTS, STYLE_PROMPTS,
    data_path, topics_path, training_model_dir,
)
from src.topics import route_request, topic_prompt


ROOT = Path(__file__).resolve().parents[1]


class MultilingualTests(unittest.TestCase):
    """Train and chat must agree on language-specific prompts and topic IDs."""

    def read_topics(self, language):
        path = ROOT / topics_path(language)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_supported_languages_and_paths(self):
        self.assertEqual(LANGUAGES, ("en", "fr", "vi"))
        self.assertEqual(str(data_path("en")), "data/jokes_clean_en.jsonl")
        self.assertEqual(str(data_path("fr")), "data/jokes_clean_fr.jsonl")
        self.assertEqual(str(topics_path("vi")), "data/topics_vi.json")

    def test_localized_taxonomies_have_same_topic_ids(self):
        english = self.read_topics("en")["topic_keyword_patterns"]
        for language in ("fr", "vi"):
            localized = self.read_topics(language)
            self.assertEqual(set(english), set(localized["topic_keyword_patterns"]))
            self.assertEqual(set(english), set(localized["topic_labels"]))
            self.assertTrue(all(localized["topic_keyword_patterns"].values()))

    def test_joke_dataset_schema_and_source_indices(self):
        for language in ("fr", "vi"):
            rows = [json.loads(line) for line in
                    (ROOT / data_path(language)).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 69)  # pilot data; not the entire English set
            self.assertEqual(len({row["source_index"] for row in rows}), len(rows))
            self.assertTrue(all(isinstance(row["joke"], str) and row["joke"]
                                and isinstance(row["topics"], list)
                                and row["style"] in STYLE_PROMPTS[language]
                                for row in rows))

    def test_french_topic_and_style_requests(self):
        t = self.read_topics("fr")
        patterns, labels = t["topic_keyword_patterns"], t["topic_labels"]
        result = route_request("Une blague sur les professeurs", patterns, "fr", labels)
        self.assertEqual(result.topic, "school_education")
        self.assertEqual(result.prompt, topic_prompt("school_education", "fr", labels))
        self.assertEqual(route_request("Un calembour", patterns, "fr", labels).prompt,
                         STYLE_PROMPTS["fr"]["pun"])
        self.assertEqual(route_request("Raconte une blague", patterns, "fr", labels).prompt,
                         GENERAL_PROMPTS["fr"])

    def test_vietnamese_topic_and_style_requests(self):
        t = self.read_topics("vi")
        patterns, labels = t["topic_keyword_patterns"], t["topic_labels"]
        self.assertEqual(route_request("Kể chuyện cười về mèo", patterns, "vi", labels).topic,
                         "animals_pets")
        self.assertEqual(route_request("Kể chuyện cười về lập trình viên",
                                       patterns, "vi", labels).topic, "programming")
        self.assertEqual(route_request("Kể chuyện cười chơi chữ",
                                       patterns, "vi", labels).prompt, STYLE_PROMPTS["vi"]["pun"])
        self.assertEqual(route_request("Kể chuyện cười", patterns, "vi", labels).prompt,
                         GENERAL_PROMPTS["vi"])

    def test_english_backward_compatible_prompts(self):
        english = self.read_topics("en")["topic_keyword_patterns"]
        self.assertEqual(route_request("Tell me a joke", english).prompt, "Tell me a joke.")
        self.assertEqual(route_request("Jokes about programmers", english).topic, "programming")


if __name__ == "__main__":
    unittest.main()
