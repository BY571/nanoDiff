"""Data loading.

Tokens are stored as a flat uint16 array on disk (see scripts/prepare_data.py) and
memory-mapped so datasets larger than RAM are fine.

A training example is simply a random contiguous window of `block_size` tokens.
Note there is NO input/target offset like in an autoregressive loader: the
diffusion loss corrupts the window itself, so the window IS both input and target.
"""
import os

import numpy as np
import torch


class TokenDataset:
    """Memory-mapped uint16 token shard; serves random fixed-length windows."""

    def __init__(self, data_path, block_size):
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"{data_path} not found — run scripts/prepare_data.py first."
            )
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) <= block_size:
            raise ValueError(
                f"dataset ({len(self.data)} tokens) is shorter than block_size "
                f"({block_size})"
            )

    def __len__(self):
        return len(self.data)

    def get_batch(self, batch_size, device):
        """Return a (batch_size, block_size) LongTensor of clean token ids (x0)."""
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64))
            for i in ix
        ])
        if str(device).startswith("cuda"):
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x
