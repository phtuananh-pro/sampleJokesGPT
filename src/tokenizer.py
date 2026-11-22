"""Byte-level BPE for joke conversations and learned joke identifier tokens."""
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class JokeTokenizer:
    """One tokenizer for dataset serialization and model inference."""

    SPECIAL_TOKENS = ["<|unk|>", "<|pad|>", "<|bos|>", "<|user|>",
                      "<|assistant|>", "<|eos|>"]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def token_id(self, token: str) -> int:
        result = self.tokenizer.token_to_id(token)
        if result is None:
            raise ValueError(f"Unknown tokenizer token: {token}")
        return result

    @classmethod
    def train_from_file(cls, file_path: str | Path, vocab_size: int = 4500,
                        min_frequency: int = 2, show_progress: bool = True,
                        extra_special_tokens: list[str] | None = None):
        """Treat joke IDs atomically so the model can condition on one identity.

        A joke ID is *not* an external retrieval key. The dataset is never
        available at inference; the model must learn joke wording in weights.
        """
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        specials = cls.SPECIAL_TOKENS + list(extra_special_tokens or [])
        trainer = BpeTrainer(
            vocab_size=max(vocab_size, len(specials) + 256),
            min_frequency=min_frequency,
            special_tokens=[AddedToken(t, single_word=False, lstrip=False, rstrip=False)
                            for t in specials],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=show_progress,
        )
        tokenizer.train([str(file_path)], trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path):
        return cls(Tokenizer.from_file(str(path)))
