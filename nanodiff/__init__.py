"""nanoDiff — a minimal, hackable masked diffusion language model.

The whole "diffusion" idea fits in three pieces:
    config.py     — one dataclass of hyperparameters
    model.py      — a bidirectional LLaMA-style transformer
    diffusion.py  — the forward masking process + the training loss
    sampler.py    — the reverse process (low-confidence remasking)

See the top-level README and sota-review/ for the papers this distills.
"""
from nanodiff.config import Config
from nanodiff.model import NanoDiff
from nanodiff.diffusion import forward_process, diffusion_loss
from nanodiff.sampler import generate

__all__ = ["Config", "NanoDiff", "forward_process", "diffusion_loss", "generate"]
