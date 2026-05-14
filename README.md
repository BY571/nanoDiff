# nanoDiff

A minimal, clean, hackable implementation of a **state-of-the-art diffusion
language model** — built to *learn, understand, train, and improve* dLLMs.

Think of it as [nanoGPT](https://github.com/karpathy/nanoGPT) / nanochat, but for
**diffusion** language models instead of autoregressive ones. It distills the
**LLaDA** recipe (the simplest formulation that has scaled to 8B–100B) down to a
small modular package you can read in an afternoon.

> The literature this distills — the lineage from D3PM to LLaDA 2.0 — is listed
> under [References](#references) at the bottom.

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

## Repo layout

```
nanodiff/
  config.py          # one dataclass — all hyperparameters
  model.py           # bidirectional LLaMA-style transformer (the "denoiser")
  diffusion.py       # forward masking process + the 1/t-weighted loss
  sampler.py         # reverse process: low-confidence remasking + semi-AR blocks
  data.py            # memory-mapped uint16 token loader
  utils.py           # WSD lr schedule, checkpoint IO, config loading
  sft.py             # [stub] supervised fine-tuning            (milestone)
  block_diffusion.py # [stub] block-diffusion training (BD3-LM) (milestone)
  ar_init.py         # [stub] init from a pretrained AR model   (milestone)

configs/
  train_150m.py      # ~150M params, single GPU
  train_1b.py        # ~1.3B params, multi-GPU / DGX-Spark

scripts/
  prepare_data.py    # stream + tokenize FineWeb-Edu -> train.bin / val.bin

train.py             # training loop (DDP, bf16, grad-accum, torch.compile)
sample.py            # generate text from a checkpoint
eval.py              # NLL-bound / perplexity on held-out data
```

The split is deliberate: `diffusion.py` and `sampler.py` are the *only* files that
contain diffusion-specific logic. Everything else is a normal transformer setup.

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

## Roadmap — the learning path

The core (pretrain + sample + eval) is fully implemented. The next milestones are
**documented stubs** — each file explains the recipe and points at the exact paper
section, so implementing them is a guided exercise:

| Milestone | File | What it adds | Paper |
|---|---|---|---|
| **SFT** | `nanodiff/sft.py` | instruction-following: mask only the response | LLaDA §2.3, Alg. 2 |
| **Block diffusion** | `nanodiff/block_diffusion.py` | block-causal *training* → KV-cache, any-length | BD3-LM (ICLR'25) |
| **AR-init** | `nanodiff/ar_init.py` | warm-start from a pretrained AR model | Dream 7B, LLaDA 2.0 |

Further ideas for the "...and maybe improve it" part: low-variance NLL estimator
(LLaDA Eq. 14), classifier-free guidance, soft-masking (IBM 2025), confidence-based
token editing (LLaDA 2.1), few-step distillation.

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

## Credits

The recipe is **LLaDA** (Nie et al. 2025), which itself builds on **MDLM**,
**SEDD**, and **D3PM**. The engineering style (single config dataclass, memmap
data, `torch.compile`/DDP loop) follows Andrej Karpathy's nanoGPT.
