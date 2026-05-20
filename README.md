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
uv sync                       # create .venv, install deps + the nanodiff package
source .venv/bin/activate     # (or prefix the commands below with `uv run`)
python tests/smoke_test.py    # optional: verify the core stack works (~2 min, CPU)

# 1. tokenize a pretraining corpus (streams FineWeb-Edu)
python scripts/prepare_data.py --out-dir data/fineweb_edu --num-tokens 2_000_000_000

# 2. pretrain a base model (single GPU)
python pretrain/train.py --config pretrain/configs/50m.py
#    ...or multi-GPU:
torchrun --standalone --nproc_per_node=8 pretrain/train.py --config pretrain/configs/50m.py

# 3. sample / evaluate
python sample.py --ckpt checkpoints/50m/ckpt.pt --prompt "The meaning of life is"
python eval.py --ckpt checkpoints/50m/ckpt.pt --iters 500

# 4. (optional) instruction-tune the base on Alpaca-cleaned
python scripts/prepare_sft_data.py --out-dir data/alpaca_sft
python sft/train.py --config sft/configs/50m_alpaca.py
```

### Scaling

Scaling is a one-file change — copy a config and edit the model/optimizer fields:

```python
# pretrain/configs/350m.py
from nanodiff.config import Config
config = Config(name="nanodiff-350m", n_layer=16, n_embd=1280, n_head=20,
                batch_size=16, grad_accum_steps=16, out_dir="checkpoints/350m")
```

Everything reads from the `Config` dataclass, so model code never changes.

---

## Pretrained models

Two pretrained checkpoints are on the Hugging Face Hub:

| Model | What it is |
|---|---|
| [Sebasdi/nanodiff-50m-base](https://huggingface.co/Sebasdi/nanodiff-50m-base) | the 50M base — pretrained on ~2B tokens of FineWeb-Edu (val perplexity ~50) |
| [Sebasdi/nanodiff-50m-sft-alpaca](https://huggingface.co/Sebasdi/nanodiff-50m-sft-alpaca) | the base, instruction-tuned on Alpaca-cleaned (~51k examples) |

```bash
# base model — continues text, document-style
hf download Sebasdi/nanodiff-50m-base nanodiff-50m-base.pt --local-dir checkpoints/
python chat.py --ckpt checkpoints/nanodiff-50m-base.pt

# SFT model — follows instructions (note the --sft flag)
hf download Sebasdi/nanodiff-50m-sft-alpaca nanodiff-50m-sft-alpaca.pt --local-dir checkpoints/
python chat.py --ckpt checkpoints/nanodiff-50m-sft-alpaca.pt --sft
```

> ⚠️ **Set your expectations.** These are **50M-parameter models trained on
> ~2B tokens** — on the order of 1/100th the data a model like GPT-2 saw. They
> are *learning artifacts*, not usable assistants:
>
> - The **base** model *continues* text — prompt it document-style
>   (`"The history of Rome is"`), not question-style.
> - The **SFT** model follows instructions (`chat.py --sft`), but at 50M params
>   it **confabulates freely** — fluent English, unreliable facts. SFT taught
>   it to *answer*, not to *know*.
> - Both **need the repetition penalty.** Small diffusion LMs collapse into
>   repetition loops under the default sampler; `chat.py` and `sample.py` enable
>   a frequency repetition penalty (`--rep-penalty 3.0`) by default.
>
> The next steps in this learning project are to **scale** — more training
> data and larger models — which is what these small-scale runs were
> calibrating for.

---

## How the training step works

The entire learning signal, from `pretrain/train.py`:

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

## Training customization

**Dataset — fully swappable.** The pipeline only ever sees a flat `uint16` token
array on disk, so it is dataset-agnostic. Either point `prepare_data.py` at any
Hugging Face text dataset (it just needs a `"text"` field):

```bash
python scripts/prepare_data.py --dataset <hf-name> --subset <config> --out-dir data/mine
```

or produce your own `train.bin` / `val.bin` (any `uint16` token dump) and set
`data_dir` in your config — the model never knows the difference.

**Tokenizer — coupled, but in known places.** The GPT-2 BPE is wired in as the
default working path. Swapping it means updating these spots:

| File(s) | What to change |
|---|---|
| `scripts/prepare_data.py`, `sample.py`, `pretrain/train.py` | `tiktoken.get_encoding("gpt2")` |
| `nanodiff/config.py` | `vocab_size`, `mask_token_id` (= last real id + 1, then pad) |
| `scripts/prepare_data.py`, `sample.py`, `pretrain/train.py` | `EOT` — the document-separator id |
| `nanodiff/data.py` | `uint16` dtype caps the vocab at 65536; use `uint32` above that |

**Model size — a one-file config change.** See [Scaling](#scaling) above.

---

## References

The recipe `nanoDiff` implements is **LLaDA**; here is the lineage:

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
