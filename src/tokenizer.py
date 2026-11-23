"""Byte-level BPE for joke conversations and learned joke identifier tokens."""
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class JokeTokenizer:
    """Train and reuse the exact same byte-level BPE tokenizer.

    Chat/control markers and per-joke IDs remain atomic vocabulary entries;
    the rest of the joke is learned as subword/byte-level tokens.
    """

    # The model learns chat boundaries as special tokens, not literal character
    # fragments. EOS teaches when an assistant has finished one joke.
    SPECIAL_TOKENS = ["<|unk|>", "<|pad|>", "<|bos|>", "<|user|>",
                      "<|assistant|>", "<|eos|>"]

    def __init__(self, tokenizer: Tokenizer):
        """Wrap the Hugging Face tokenizer used by both train and chat."""
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        """Actual learned vocabulary size, including special joke-ID tokens."""
        return self.tokenizer.get_vocab_size()

    def token_id(self, token: str) -> int:
        """Resolve one dedicated control/identity token without re-encoding."""
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
        # Byte-level preprocessing makes arbitrary UTF-8 joke text encodable;
        # BPE then merges frequent adjacent sequences into useful subwords.
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        # Give each curated joke its own atomic identity for neural generation;
        # vocabulary capacity must leave room for the 256 byte alphabet too.
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
        """Map serialized conversation text to token IDs (Transformer input)."""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """Reconstruct generated joke text from model-predicted token IDs."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        """Persist the exact BPE merges and special tokens beside model weights."""
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path):
        """Reload the identical tokenizer for inference without the dataset."""
        return cls(Tokenizer.from_file(str(path)))
