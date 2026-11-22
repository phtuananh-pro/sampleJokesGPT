"""Small, fast unit tests for data preparation and masked GPT supervision.

Run from the repository root with:
    python -m unittest discover -s tests -v

These tests do not train the model or establish joke quality.
"""

import unittest

import torch

from src.topics import detect_requested_topic, detect_topics
from src.training_data import (
    Conversation,
    build_prefix,
    encode_conversations,
    parse_conversations,
    sample_batch,
    split_by_joke,
)


class CharacterTokenizer:
    """Deterministic fake tokenizer: enough to check token-alignment logic."""

    def encode(self, value):
        return [ord(char) + 1 for char in value]


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_parse_complete_conversations(self):
        text = (
            "<|bos|>\n<|user|>Tell me a joke.\n<|assistant|>\n"
            "Why did the chicken cross the road? To get to the other side.\n<|eos|>\n\n"
            "<|bos|>\n<|user|>Tell me a programming joke.\n<|assistant|>\n"
            "Why did the programmer quit? No cache.\n<|eos|>\n"
        )
        examples = parse_conversations(text)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].request, "Tell me a joke.")
        self.assertTrue(examples[1].joke.endswith("No cache."))

    def test_split_keeps_prompt_variants_in_one_partition(self):
        examples = [
            Conversation("Tell me a joke.", "Joke A"),
            Conversation("Tell me a programming joke.", "Joke A"),
            Conversation("Tell me a joke.", "Joke B"),
            Conversation("Tell me a joke.", "Joke C"),
        ]
        train, val = split_by_joke(examples, seed=7)
        self.assertTrue(train and val)
        self.assertFalse({ex.joke for ex in train} & {ex.joke for ex in val})

    def test_assistant_only_loss_masks_prompt_and_padding(self):
        request = "Tell me a programming joke."
        example = Conversation(request, "Programmers love coffee.")
        encoded, skipped = encode_conversations([example], self.tokenizer, block_size=200)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(encoded), 1)
        sample = encoded[0]
        x, y = sample_batch(encoded, 1, 200, pad_id=0, device="cpu")
        first_answer_position = len(self.tokenizer.encode(build_prefix(request))) - 1
        self.assertTrue(torch.all(y[0, :first_answer_position] == -100))
        self.assertEqual(
            y[0, first_answer_position].item(),
            self.tokenizer.encode("Programmers love coffee.\n<|eos|>")[0],
        )
        self.assertTrue(torch.all(y[0, len(sample.tokens) - 1:] == -100))
        self.assertTrue(torch.all(x[0, len(sample.tokens) - 1:] == 0))
        self.assertGreater(int((y != -100).sum()), 0)

    def test_overlong_examples_skipped(self):
        examples, skipped = encode_conversations(
            [Conversation("Tell me a joke.", "A" * 500)], self.tokenizer, block_size=128
        )
        self.assertEqual(examples, [])
        self.assertEqual(skipped, 1)

    def test_topic_routing(self):
        self.assertEqual(detect_requested_topic("joke on programming"), "programming")
        self.assertEqual(detect_requested_topic("Tell me a joke about animals"), "animal")
        self.assertEqual(detect_requested_topic("joke about children"), "children")
        self.assertIsNone(detect_requested_topic("Tell me a joke"))
        # A snake in an animal joke must not, by itself, label it programming.
        self.assertNotIn("programming", detect_topics("An elephant met a python."))


if __name__ == "__main__":
    unittest.main()
