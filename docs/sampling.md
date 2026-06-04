# How sampling works

A short walkthrough of the iterative un-masking sampler used by
`chat.py`, `sample.py`, and the periodic sanity-sample in
`pretrain/train.py`. The full implementation is in
[`nanodiff/sampler.py`](../nanodiff/sampler.py).

## The core loop

Generation starts with the prompt followed by `gen_length` MASK tokens,
then iterates: predict every position, commit the most confident ones,
re-mask the rest, repeat.

```python
x = prompt + [MASK] * gen_length
for step in range(steps):
    logits = model(x)                       # predict every position at once
    conf   = softmax(logits).max(-1)        # per-position certainty
    # commit the K most-confident currently-masked positions:
    x[topk(conf masked to MASK positions, K)] = argmax(logits)[those positions]
```

The schedule sets K: with `steps == gen_length` (the default), K=1 (one
token committed per step). With `steps=32, gen_length=96`, K≈3 per
step. With `steps=16, gen_length=96`, K=6 per step. Fewer steps means
the model commits more tokens in parallel each iteration, trading
quality for speed.

## The `block_length` knob

`block_length` splits the generation into semi-autoregressive chunks
that get filled in sequence:

| `block_length` setting | Behaviour |
|---|---|
| `block_length == gen_length` | Pure diffusion; the model fills all `gen_length` positions in parallel across `steps` iterations |
| `block_length == 1` | Strict left-to-right autoregressive; each position is committed before the next is even considered |
| `block_length == gen_length // n` | Semi-autoregressive in `n` blocks; each block is filled diffusion-style, blocks are committed in order |

The chat default (`block_length=32, gen_length=96`) is three semi-AR
blocks of 32 tokens each. Each block gets `steps / n_blocks` denoising
iterations, with K commits distributed across them.

## The within-step rep_penalty

At small `steps`, two positions committed in the same step can
independently agree on the same token ("process process", "the the"),
because the standard `rep_penalty` only sees tokens committed in
*prior* steps. The sampler counters this with a **within-step
rep_penalty**: after sampling a candidate token at each masked
position, same-token collisions inside the step are broken by
penalising the loser's confidence so it falls out of the top-K commit
set.

That fix is what makes `--steps 32` (the validated fast preset) safe.
Below `--steps 24` some doubling reappears because the base
`rep_penalty` takes a few steps to build up against a strongly-favored
repeating token.

## `--steps` ≠ `max_iters`

Worth flagging because both are called "steps" in different parts of
nanoDiff:

- `--steps` in `chat.py` / `sample.py` (or `cfg.sample_steps` in
  training configs) counts **denoising iterations within one
  generation**. Range: 16-96 in our typical use. Tunable per call.
- `max_iters` in `pretrain/configs/*.py` and `sft/configs/*.py` counts
  **optimizer iterations across the dataset**. Range: 5,000-76,300 in
  our family runs. Fixed per training run.

Same word, different loop. The training loop's "step" advances by one
optimizer update; the sampling loop's "step" advances by one denoising
pass over a single generation.

## See also

- [`nanodiff/sampler.py`](../nanodiff/sampler.py) — the production
  implementation, with the full set of knobs (top_p, top_k,
  rep_penalty, temperature, tau threshold decoding, use_cache, etc.)
- [README → Sampling speed](../README.md#sampling-speed) — the
  user-facing flags and their measured speed trade-offs
- [LLaDA paper](https://arxiv.org/abs/2502.09992) — the algorithm
  nanoDiff implements (Algorithm 1 for the loop, Algorithm 5 for the
  low-confidence remasking strategy)
