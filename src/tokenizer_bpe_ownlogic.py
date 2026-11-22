import json
import re
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """Learn byte-pair merges and map UTF-8 text to reversible integer IDs."""

    SPECIAL_TOKENS = [
        "<|pad|>",
        "<|bos|>",
        "<|user|>",
        "<|assistant|>",
        "<|eos|>",
    ]

    def __init__(self):
        """Initialize special IDs, all byte IDs, and empty merge tables."""
        self.special_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}

        offset = len(self.SPECIAL_TOKENS)

        self.byte_to_id = {b: offset + b for b in range(256)}
        self.id_to_bytes = {offset + b: bytes([b]) for b in range(256)}

        self.merges = []
        self.merge_ranks = {}

        self.vocab_size = offset + 256

        special_pattern = "|".join(re.escape(tok) for tok in self.SPECIAL_TOKENS)
        self.special_re = re.compile(f"({special_pattern})")

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 2000,
        min_pair_freq: int = 2,
        verbose: bool = True,
    ):
        """Learn frequent adjacent byte pairs until the target vocabulary size.

        Each selected pair is assigned a new ID and replaced throughout the
        training sequences, which is the iterative BPE vocabulary-building
        procedure.
        """
        tokenizer = cls()

        target_vocab_size = max(vocab_size, tokenizer.vocab_size)

        sequences = tokenizer._text_to_initial_sequences(text)

        while tokenizer.vocab_size < target_vocab_size:
            pair_counts = Counter()

            for seq in sequences:
                for a, b in zip(seq, seq[1:]):
                    if tokenizer._is_special_id(a) or tokenizer._is_special_id(b):
                        continue
                    pair_counts[(a, b)] += 1

            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]

            if best_count < min_pair_freq:
                break

            new_id = tokenizer.vocab_size
            a, b = best_pair

            tokenizer.id_to_bytes[new_id] = tokenizer.id_to_bytes[a] + tokenizer.id_to_bytes[b]
            tokenizer.merges.append([a, b, new_id])
            tokenizer.merge_ranks[(a, b)] = (len(tokenizer.merges) - 1, new_id)
            tokenizer.vocab_size += 1

            sequences = [
                tokenizer._replace_pair(seq, best_pair, new_id)
                for seq in sequences
            ]

            if verbose and tokenizer.vocab_size % 100 == 0:
                print(f"BPE vocab size: {tokenizer.vocab_size}, last pair freq: {best_count}")

        return tokenizer

    def _is_special_id(self, token_id: int) -> bool:
        """Return whether an ID is reserved for a chat/control marker."""
        return token_id in self.id_to_special

    def _text_to_initial_sequences(self, text: str) -> list[list[int]]:
        """Convert each input line independently to initial byte/special IDs."""
        # Keep lines separate to avoid learning huge cross-line merges.
        lines = text.splitlines(keepends=True)
        sequences = []

        for line in lines:
            ids = self._initial_encode(line)
            if ids:
                sequences.append(ids)

        return sequences

    def _initial_encode(self, text: str) -> list[int]:
        """Split recognized special markers from text and UTF-8 encode the rest."""
        ids = []

        parts = self.special_re.split(text)

        for part in parts:
            if not part:
                continue

            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                for b in part.encode("utf-8"):
                    ids.append(self.byte_to_id[b])

        return ids

    @staticmethod
    def _replace_pair(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace non-overlapping occurrences of a pair with its merged ID."""
        out = []
        i = 0

        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1

        return out

    def encode(self, text: str) -> list[int]:
        """Encode text with the merges learned by this tokenizer instance."""
        sequences = self._text_to_initial_sequences(text)
        encoded = []

        for seq in sequences:
            encoded.extend(self._encode_sequence(seq))

        return encoded

    def _encode_sequence(self, seq: list[int]) -> list[int]:
        """Apply available merges in learned rank order to one token sequence."""
        seq = list(seq)

        while True:
            best_rank = None
            best_pair = None
            best_new_id = None

            for a, b in zip(seq, seq[1:]):
                found = self.merge_ranks.get((a, b))

                if found is None:
                    continue

                rank, new_id = found

                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = (a, b)
                    best_new_id = new_id

            if best_pair is None:
                break

            seq = self._replace_pair(seq, best_pair, best_new_id)

        return seq

    def decode(self, ids: list[int]) -> str:
        """Reassemble bytes and special markers, replacing invalid UTF-8 safely."""
        parts = []
        byte_buffer = bytearray()

        def flush_bytes():
            """Decode the buffered ordinary bytes before emitting a marker."""
            nonlocal byte_buffer
            if byte_buffer:
                parts.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer = bytearray()

        for token_id in ids:
            token_id = int(token_id)

            if token_id in self.id_to_special:
                flush_bytes()
                parts.append(self.id_to_special[token_id])
            elif token_id in self.id_to_bytes:
                byte_buffer.extend(self.id_to_bytes[token_id])
            else:
                flush_bytes()
                parts.append("�")

        flush_bytes()
        return "".join(parts)

    def save(self, path: str | Path):
        """Persist special tokens, merges, and byte expansions as JSON."""
        path = Path(path)

        data = {
            "special_tokens": self.SPECIAL_TOKENS,
            "merges": self.merges,
            "id_to_bytes": {
                str(k): v.hex()
                for k, v in self.id_to_bytes.items()
            },
            "vocab_size": self.vocab_size,
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path):
        """Restore merge ranks and byte expansions from a saved tokenizer."""
        path = Path(path)

        data = json.loads(path.read_text(encoding="utf-8"))

        tokenizer = cls()

        tokenizer.merges = data["merges"]
        tokenizer.id_to_bytes = {
            int(k): bytes.fromhex(v)
            for k, v in data["id_to_bytes"].items()
        }

        tokenizer.vocab_size = data["vocab_size"]

        tokenizer.merge_ranks = {}

        for rank, item in enumerate(tokenizer.merges):
            a, b, new_id = item
            tokenizer.merge_ranks[(a, b)] = (rank, new_id)

        return tokenizer
"""Legacy from-scratch byte-level BPE tokenizer kept for study.

The runnable chatbot uses :mod:`src.tokenizer` and the ``tokenizers`` package.
This separate implementation exposes the merge-counting and byte-encoding
ideas but is not used by the training or chat entry points.
"""

