"""Small shared helpers: config loading, LR schedule, checkpoint IO."""
import importlib.util
import os

import torch


def load_config(path):
    """Load the `config` object defined in a python file under configs/."""
    spec = importlib.util.spec_from_file_location("nanodiff_run_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise AttributeError(f"{path} must define a top-level `config` object")
    return module.config


def unwrap_model(model):
    """Strip DistributedDataParallel and torch.compile wrappers -> raw nn.Module."""
    if hasattr(model, "module"):        # DistributedDataParallel
        model = model.module
    if hasattr(model, "_orig_mod"):     # torch.compile
        model = model._orig_mod
    return model


def get_lr(it, cfg):
    """Warmup-Stable-Decay (WSD) learning-rate schedule — LLaDA's choice.

        warmup: linear   0  -> lr        over the first  `warmup_iters`
        stable: constant lr               in the middle
        decay : linear   lr -> min_lr     over the final  `decay_iters`

    WSD keeps the LR flat for most of training, so you can stop (and decay) at any
    point without the cosine schedule's "you must commit to max_iters up front".
    """
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    decay_start = cfg.max_iters - cfg.decay_iters
    if it >= decay_start:
        progress = min((it - decay_start) / max(1, cfg.decay_iters), 1.0)
        return cfg.lr - progress * (cfg.lr - cfg.min_lr)
    return cfg.lr


def save_checkpoint(path, model, optimizer, iter_num, cfg, best_val=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = unwrap_model(model)
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "iter_num": iter_num,
        "config": cfg,
        "best_val": best_val,
    }, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def human(n):
    """Format a count like 1_234_567 -> '1.2M'."""
    n = float(n)
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"
