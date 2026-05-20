"""Configuration for nanoDiff.

A single dataclass holds model, diffusion, data, optimizer and IO settings.
Concrete experiments live in `configs/` and just instantiate this with overrides,
so scaling from 150M to 1B is a one-file change:

    configs/train_150m.py   ->   configs/train_1b.py
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # ---- run identity ----
    name: str = "nanodiff"

    # ---- model (LLaMA-style transformer, but BIDIRECTIONAL) ----
    # GPT-2 BPE has 50257 tokens. We append one [MASK] token (the absorbing state,
    # id 50257) and pad the vocab to a multiple of 64 for kernel efficiency.
    vocab_size: int = 50304
    mask_token_id: int = 50257
    block_size: int = 1024          # max sequence length
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    mlp_ratio: float = 4.0          # SwiGLU hidden ~= mlp_ratio * n_embd * 2/3
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    # ---- diffusion ----
    # t ~ U(t_eps, 1); the lower clamp keeps the 1/t loss weight finite.
    t_eps: float = 1e-3

    # ---- data ----
    data_dir: str = "data/fineweb_edu"   # expects train.bin / val.bin (uint16 tokens)

    # ---- optimization ----
    batch_size: int = 32            # micro-batch per device
    grad_accum_steps: int = 8       # effective batch = batch_size * grad_accum * world_size
    max_iters: int = 100_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # Warmup-Stable-Decay (WSD) schedule, as used by LLaDA.
    lr: float = 4e-4
    min_lr: float = 1e-5
    schedule: str = "wsd"           # "wsd" | "cosine"
    warmup_iters: int = 2000
    # WSD only: linear decay over the FINAL `decay_iters` steps. The 50M v2
    # schedule sweep (2026-05) found long decay wins: ~0.6 * max_iters beat
    # both the short 0.2 default and pure cosine. Rule of thumb: decay ~= 60%.
    decay_iters: int = 60_000

    # ---- evaluation / sampling during training ----
    eval_interval: int = 1000
    eval_iters: int = 100
    log_interval: int = 10
    sample_interval: int = 2000
    sample_steps: int = 128         # denoising steps for the periodic sanity sample
    sample_gen_length: int = 128

    # ---- system ----
    device: str = "cuda"
    dtype: str = "bfloat16"         # "bfloat16" | "float16" | "float32"
    compile: bool = True
    compile_mode: str = "default"   # "default" | "reduce-overhead" (cuda graphs) | "max-autotune"

    # ---- io ----
    out_dir: str = "checkpoints/nanodiff"

    # ---- logging ----
    wandb_log: bool = False         # set True to log metrics to Weights & Biases
    wandb_project: str = "nanodiff"

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        head_dim = self.n_embd // self.n_head
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.mask_token_id < self.vocab_size, "mask_token_id must fit in vocab"
        assert self.warmup_iters + self.decay_iters <= self.max_iters, (
            "warmup_iters + decay_iters cannot exceed max_iters"
        )
