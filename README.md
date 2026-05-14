# nanoDiff

A minimal, clean, hackable implementation of a **state-of-the-art diffusion
language model** — built to *learn, understand, train, and improve* dLLMs.

Think of it as [nanoGPT](https://github.com/karpathy/nanoGPT) / nanochat, but for
**diffusion** language models instead of autoregressive ones. It distills the
**LLaDA** recipe (the simplest formulation that has scaled to 8B–100B) down to a
small modular package you can read in an afternoon.

> Based on **LLaDA — Large Language Diffusion Models**
> ([Nie et al. 2025, arXiv:2502.09992](https://arxiv.org/abs/2502.09992)). The
> broader lineage (D3PM → LLaDA 2.0) is in [References](#references).

---

## The idea in 60 seconds

An autoregressive LLM writes text left-to-right: `p(x) = Π p(xₜ | x<ₜ)`.

A **masked diffusion LM** instead learns to *un-corrupt* text:

- **Forward process** (`nanodiff/diffusion.py`): pick a mask ratio `t ~ U(0,1)`,
  then replace each token independently with a `[MASK]` token with probability
  `t`. That's the entire "noising" process — no Gaussians, no latents.
- **Model** (`nanodiff/model.py`): a LLaMA-style transformer (RMSNorm, SwiGLU,
  RoPE) with **one change** — attention is *bidirectional*, so it can use
  right-context to fill in masks.
- **Loss** (`nanodiff/diffusion.py`): cross-entropy on the masked positions only,
  weighted by `1/t`. That weight is what makes the loss a real upper bound on the
  negative log-likelihood (not just a heuristic).
- **Sampling** (`nanodiff/sampler.py`): start fully masked, repeatedly predict all
  tokens but only **commit the most confident ones**, re-mask the rest, iterate.
  A `block_length` knob smoothly interpolates between pure-diffusion and
  autoregressive decoding.

Why care: parallel generation, bidirectional context, native infilling, and a
tunable quality↔speed dial.

```
        t=1  ████████████████████   (all [MASK])
             ██ the ███ of ████ is
             ██ the meaning of life is
        t=0  the meaning of life is   (clean text)
                ▲ each step: predict all, commit the confident ones, repeat
```

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. tokenize a corpus (streams FineWeb-Edu; ~2B tokens is plenty for the 150M)
python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000

# 2. train (single GPU)
python train.py --config configs/train_150m.py

#    ...or multi-GPU / DGX-Spark
torchrun --standalone --nproc_per_node=8 train.py --config configs/train_1b.py

# 3. sample
python sample.py --ckpt checkpoints/150m/ckpt.pt --prompt "The meaning of life is"

# 4. evaluate
python eval.py --ckpt checkpoints/150m/ckpt.pt --iters 500
```

### Scaling

Scaling is a one-file change — copy a config and edit the model/optimizer fields:

```python
# configs/train_350m.py
from nanodiff.config import Config
config = Config(name="nanodiff-350m", n_layer=16, n_embd=1280, n_head=20,
                batch_size=16, grad_accum_steps=16, out_dir="checkpoints/350m")
```

Everything reads from the `Config` dataclass, so model code never changes.

---

## How the training step works

The entire learning signal, from `train.py`:

```python
x0           = train_data.get_batch(...)               # clean tokens  (B, T)
x_t, mask, t = forward_process(x0, mask_token_id)      # corrupt them
logits       = model(x_t)                              # predict every token
loss         = diffusion_loss(logits, x0, mask, t)     # 1/t-weighted CE on masks
loss.backward()
```

That's it. No noise schedule, no timestep embedding (we use LLaDA's *time-free*
parameterization — see the comment at the top of `model.py`), no ELBO bookkeeping.

---

## References

The recipe `nanoDiff` implements is **LLaDA**; the lineage and the papers each
stub points to:

| Paper | Year | arXiv |
|---|---|---|
| D3PM — Structured Denoising Diffusion in Discrete State-Spaces | 2021 | [2107.03006](https://arxiv.org/abs/2107.03006) |
| SEDD — Discrete Diffusion by Estimating Data-Distribution Ratios | 2024 | [2310.16834](https://arxiv.org/abs/2310.16834) |
| MDLM — Simple and Effective Masked Diffusion Language Models | 2024 | [2406.07524](https://arxiv.org/abs/2406.07524) |
| BD3-LM — Block Diffusion (interpolating AR ↔ diffusion) | 2025 | [2503.09573](https://arxiv.org/abs/2503.09573) |
| **LLaDA — Large Language Diffusion Models** (primary reference) | 2025 | [2502.09992](https://arxiv.org/abs/2502.09992) |
| Dream 7B — Diffusion Large Language Models | 2025 | [2508.15487](https://arxiv.org/abs/2508.15487) |
| LLaDA 2.0 — Scaling Diffusion Language Models to 100B | 2025 | [2512.15745](https://arxiv.org/abs/2512.15745) |
| A Survey on Diffusion Language Models | 2025 | [2508.10875](https://arxiv.org/abs/2508.10875) |
