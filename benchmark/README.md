# Benchmarks

Task evaluation for nanoDiff models. Perplexity (`eval.py`, repo root) measures
how well a model fits held-out text; the benchmarks here measure whether that
translates into a concrete *capability*.

## LAMBADA

[LAMBADA](https://arxiv.org/abs/1606.06031) (Paperno et al. 2016) — predict the
**last word** of a passage. The passages are filtered so the final word follows
from the broad discourse but not from the last sentence alone, so it genuinely
tests long-range understanding.

```bash
python benchmark/lambada.py --ckpt checkpoints/50m/ckpt.pt
python benchmark/lambada.py --ckpt <ckpt> --limit 200   # quick subset
```

Reports two numbers:
- **accuracy** — fraction of passages where the model's predicted last word is
  exactly correct
- **perplexity** — `exp(mean cross-entropy)` over the last-word tokens

### Why LAMBADA (and only LAMBADA, for now)

The standard multiple-choice benchmarks — HellaSwag, MMLU, ARC — sit at random
chance (25-50%) until a model is *far* larger than nanoDiff's 50-150M. Running
them now would report pure noise.

LAMBADA last-word prediction, by contrast, has real signal even at this scale,
so it tracks genuine progress across the data/model scaling experiments. **When
LAMBADA accuracy saturates, that is the cue to add the harder benchmarks** —
MMLU, HellaSwag, ARC, GSM8K.

### Scoring a diffusion LM

A masked diffusion LM has no autoregressive chain rule, so likelihoods are
scored differently from a standard (AR) model — this is why lm-evaluation-
harness does not work out of the box.

For LAMBADA: lay down `[context | last-word]`, mask the last-word tokens, run a
single forward pass, and read the model's predictions at the masked positions.
Accuracy is an exact argmax match on every last-word token; perplexity is the
cross-entropy of the target tokens. This is the *single-pass* variant (the
whole word masked at once) — deterministic, and consistent for comparing
checkpoints.
