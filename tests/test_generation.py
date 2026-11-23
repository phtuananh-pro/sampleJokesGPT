"""Focused generation behavior tests independent of saved checkpoints.

Run: python -m unittest discover -s tests -v
"""

import unittest

import torch

from src.model import GPT


class ForcedEosModel:
    """Deterministic fake logits isolate the decoding loop from trained weights.

    Generating an actual funny joke needs trained weights; the test verifies
    the neural decoder obeys the learned EOS token and the hard length cap.
    """

    def __init__(self, eos_id: int):
        self.config = type("Config", (), {"block_size": 16})()
        self.eos_id = eos_id
        self.calls = 0

    def __call__(self, token_ids):
        self.calls += 1
        logits = torch.full((1, token_ids.shape[1], 8), -1000.0)
        logits[:, -1, self.eos_id] = 1000.0
        return logits, None


class GenerationTests(unittest.TestCase):
    """Check structural decoding behavior without expensive model training."""

    def test_eos_ends_generation(self):
        """EOS immediately terminates rather than consuming all token budget."""
        model = ForcedEosModel(eos_id=5)
        result = GPT.generate(
            model, torch.tensor([[2, 3]]),
            max_new_tokens=10, temperature=0, eos_token_id=5,
        )
        self.assertEqual(result.tolist(), [[2, 3, 5]])
        self.assertEqual(model.calls, 1)

    def test_max_new_tokens_remains_safety_limit(self):
        """A maximum length prevents unbounded loops when EOS is absent."""
        model = ForcedEosModel(eos_id=5)
        result = GPT.generate(
            model, torch.tensor([[2, 3]]),
            max_new_tokens=3, temperature=0, eos_token_id=None,
        )
        self.assertEqual(result.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
