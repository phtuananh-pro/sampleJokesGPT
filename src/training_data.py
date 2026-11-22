"""Conversation-level training data for a small, topic-conditioned joke GPT.

The model still performs ordinary causal next-token prediction. Unlike random
windows from one giant text stream, batches contain whole user/assistant
examples. Only assistant tokens, including EOS, contribute to the loss.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

import torch


_RECORD = re.compile(
    r"<\|bos\|>\r?\n<\|user\|>([^\r\n]+)\r?\n"
    r"<\|assistant\|>\r?\n(.*?)\r?\n<\|eos\|>",
    re.DOTALL,
)


@dataclass(frozen=True)
class Conversation:
    """One human request and its complete target joke."""

    request: str
    joke: str


@dataclass(frozen=True)
class EncodedConversation:
    """Token sequence and the first token position belonging to the answer."""

    tokens: tuple[int, ...]
    prefix_length: int


def build_prefix(request: str) -> str:
    """Match the exact prefix used both at training and during chat."""

    return f"<|bos|>\n<|user|>{request}\n<|assistant|>\n"


def parse_conversations(text: str) -> list[Conversation]:
    """Read serialized records without accidentally training across joke boundaries."""

    examples = [
        Conversation(request=match.group(1).strip(), joke=match.group(2).strip())
        for match in _RECORD.finditer(text)
    ]
    examples = [ex for ex in examples if ex.request and ex.joke]
    if not examples:
        raise ValueError("No complete <|bos|>/<|user|>/<|assistant|>/<|eos|> conversations found.")
    return examples


def split_by_joke(
    examples: list[Conversation], seed: int, val_fraction: float = 0.1,
) -> tuple[list[Conversation], list[Conversation]]:
    """Keep every prompt variant of the same joke in the SAME split.

    Splitting a flat token stream leaks duplicated answers into validation when
    the cleaner writes both a general and topic-specific prompt for one joke.
    """

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    groups: dict[str, list[Conversation]] = defaultdict(list)
    for example in examples:
        key = " ".join(example.joke.casefold().split())
        groups[key].append(example)

    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two distinct jokes for an independent validation set.")
    random.Random(seed).shuffle(keys)
    n_val = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n_val])
    train, val = [], []
    for key in keys:
        (val if key in val_keys else train).extend(groups[key])
    return train, val


def encode_conversations(
    examples: list[Conversation], tokenizer, block_size: int,
) -> tuple[list[EncodedConversation], int]:
    """Encode entire prompt+answer records; skip samples too long for context.

    Tokenizing the prefix separately is intentional: inference encodes that
    exact prefix before sampling the first answer token.
    """

    accepted: list[EncodedConversation] = []
    skipped = 0
    for example in examples:
        prefix = tokenizer.encode(build_prefix(example.request))
        answer = tokenizer.encode(f"{example.joke}\n<|eos|>")
        tokens = tuple(prefix + answer)
        if len(tokens) < 2 or len(tokens) - 1 > block_size:
            skipped += 1
            continue
        accepted.append(EncodedConversation(tokens=tokens, prefix_length=len(prefix)))
    return accepted, skipped


def sample_batch(
    examples: list[EncodedConversation], batch_size: int,
    block_size: int, pad_id: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each full conversation and mask prompt/padding labels with -100.

    The first unmasked target is the first answer token, predicted by the final
    prompt token. EOS is also a target, so generation can learn when to stop.
    """

    if not examples:
        raise ValueError("Cannot sample from an empty conversation split.")
    x = torch.full((batch_size, block_size), pad_id, dtype=torch.long)
    y = torch.full((batch_size, block_size), -100, dtype=torch.long)

    for row, sample_index in enumerate(torch.randint(len(examples), (batch_size,)).tolist()):
        example = examples[sample_index]
        seq_len = len(example.tokens) - 1
        start = example.prefix_length - 1
        x[row, :seq_len] = torch.tensor(example.tokens[:-1], dtype=torch.long)
        y[row, start:seq_len] = torch.tensor(example.tokens[example.prefix_length:], dtype=torch.long)

    return x.to(device), y.to(device)
