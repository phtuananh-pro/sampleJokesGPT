"""Explicit decoder-only Transformer components used by the joke chatbot.

The tensor flow follows Part IV of the Frontier Models guide: token and
position embeddings enter repeated causal-attention/MLP blocks, then a shared
vocabulary head predicts the next token. PyTorch provides tensor operations,
autograd, and optimization; the Transformer operations remain visible here.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Store the vocabulary and dimensions needed to build a GPT model.

    Training saves these values with the checkpoint so inference can recreate
    the same architecture before loading its learned parameters.
    """

    vocab_size: int
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    """Route information among positions without exposing future tokens.

    Learned Q/K/V projections produce per-head queries, keys, and values. A
    lower-triangular mask removes future positions before softmax, and the
    resulting weights mix value vectors as described in Part IV.7–8.
    """

    def __init__(self, config: GPTConfig):
        """Create projections, dropout layers, and the fixed causal mask."""
        super().__init__()

        if config.n_head <= 0 or config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by a positive n_head.")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "mask",
            mask.view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        """Apply multi-head causal attention to ``[batch, time, width]``."""
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = q @ k.transpose(-2, -1)
        att = att / (self.head_dim ** 0.5)

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class MLP(nn.Module):
    """Apply the same nonlinear feature transform independently per token.

    Attention mixes across sequence positions; this FFN expands and contracts
    each position's feature vector, preserving the model width (Part IV.10).
    """

    def __init__(self, config: GPTConfig):
        """Build the expansion, GELU, projection, and dropout layers."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        """Transform token features while preserving batch and time axes."""
        return self.net(x)


class Block(nn.Module):
    """One pre-normalized residual attention-plus-FFN Transformer block."""

    def __init__(self, config: GPTConfig):
        """Create separate normalization, attention, and MLP sublayers."""
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """Add attention and MLP updates through residual connections."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """A small decoder-only GPT that learns to predict the next joke token."""

    def __init__(self, config: GPTConfig):
        """Build embeddings, the repeated block stack, and vocabulary head."""
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tying these matrices shares token input/output representations and
        # reduces parameters; both module names remain in the state dict.
        self.lm_head.weight = self.token_embedding.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")

    def _init_weights(self, module):
        """Initialize linear and embedding weights with a small normal scale."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Return vocabulary logits and optional shifted-token cross-entropy.

        Causal masking allows all positions to be processed together while
        each prediction can depend only on its preceding context.
        """
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} > block_size {self.config.block_size}"
            )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)

        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 0.9,
        top_k: int | None = 50,
    ):
        """Autoregressively sample tokens, cropping input to the context window.

        At each step the model predicts a distribution for the next token;
        temperature and top-k alter sampling, then the selected token is
        appended so it can condition the following prediction (Part IV.14).
        """
        if temperature < 0:
            raise ValueError("temperature must be zero or greater.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be zero or greater.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive or None.")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)

        return idx
