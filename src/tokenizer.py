from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class JokeTokenizer:
    """Train, save, and use the BPE vocabulary shared by training and chat."""

    SPECIAL_TOKENS = [
        "<|unk|>",
        "<|pad|>",
        "<|bos|>",
        "<|user|>",
        "<|assistant|>",
        "<|eos|>",
    ]

    def __init__(self, tokenizer: Tokenizer):
        """Wrap a Hugging Face ``tokenizers`` tokenizer instance."""
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the number of entries in the trained vocabulary."""
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train_from_file(
        cls,
        file_path: str | Path,
        vocab_size: int = 2000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ):
        """Train a byte-level BPE tokenizer from a text file.

        Byte-level pre-tokenization lets the vocabulary represent arbitrary
        UTF-8 joke text, while added special tokens preserve chat boundaries.
        """
        file_path = Path(file_path)

        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        special_tokens = [
            AddedToken(tok, single_word=False, lstrip=False, rstrip=False)
            for tok in cls.SPECIAL_TOKENS
        ]

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=show_progress,
        )

        tokenizer.train([str(file_path)], trainer)

        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """Convert text, including recognized control markers, to token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Convert token IDs back to text, optionally omitting control tokens."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path):
        """Write vocabulary and merge rules so later runs can reuse them."""
        path = Path(path)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path):
        """Rebuild a tokenizer wrapper from its saved JSON vocabulary."""
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)
"""Byte-level BPE tokenizer adapter for the conversation training format.

The tokenizer maps visible text and markers such as ``<|user|>`` to IDs and
back. It defines the model's vocabulary boundary; it does not perform
Transformer attention or learn the next-token objective.
"""

