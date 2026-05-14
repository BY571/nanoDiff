"""nanoDiff model: a bidirectional LLaMA-style transformer.

The ONLY architectural difference from an autoregressive GPT is that attention is
bidirectional (no causal mask): a diffusion LM must see the whole partially-masked
sequence to fill in the masked positions.

We also adopt LLaDA's *time-free* parameterization: the timestep `t` is NOT fed to
the network. Because unmasked tokens already pin down the clean data, and masked
tokens are predicted purely from context, the optimal denoiser is time-invariant
(LLaDA, Eq. 11). This deletes all the timestep-embedding machinery you would see
in an image diffusion model — the model is genuinely just a transformer.

Components are otherwise standard modern-LLM: RMSNorm, SwiGLU, RoPE, pre-norm
residual blocks.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float):
    """Precompute the rotary position embedding cos/sin tables, shape (T, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)          # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotate query/key vectors. x: (B, n_head, T, head_dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    """Multi-head self-attention — bidirectional (this is the diffusion-LM change)."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # is_causal=False  <-- the heart of a diffusion LM: every token attends to
        # every other token, so the model can use right-context to fill in [MASK]s.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (the LLaMA MLP)."""

    def __init__(self, config):
        super().__init__()
        hidden = int(config.mlp_ratio * config.n_embd * 2 / 3)
        hidden = 64 * ((hidden + 63) // 64)            # round up to a multiple of 64
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)   # gate
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)   # value
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)   # down-projection
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class NanoDiff(nn.Module):
    """The full model: token embedding -> N bidirectional blocks -> logits."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style: scale down the residual-path projections by 1/sqrt(2 * n_layer)
        # so the residual stream variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """idx: (B, T) token ids, with [MASK] at corrupted positions -> logits (B, T, V)."""
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size"
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T].view(1, 1, T, -1).to(x.dtype)
        sin = self.rope_sin[:T].view(1, 1, T, -1).to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        """AdamW with weight decay on 2D+ params only (matrices), not on norms/biases."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas,
                                 fused=(device_type == "cuda"))
