# Experiment Log

Empirical findings from each training run, in chronological order. What matters
here is the *dynamics* and what they imply about the next experiment, not the
absolute numbers (those date instantly).

All runs use the same 30M-param model (`configs/train_30m*.py`): 6 layers,
n_embd 384, n_head 6, block_size 512, GPT-2 BPE, masked-diffusion objective,
trained on FineWeb-Edu (`sample-10BT`). Training on a Jetson Orin (eager mode,
bf16 autocast, no compile, no DDP — the Jetson torch build supports none of those).

---

## Run 1 — 30M · 100M tokens · 10k iters

**Setup.** WSD warmup 300 / decay 2000. ~3.3 hrs on the Jetson.

**Result.** Best val **5.14**, perplexity **170**.

**What we learned.**
- The pipeline works end-to-end on real text — loss descends cleanly from
  `ln(vocab) = 10.83` to ~5.5 inside the first 1000 iters.
- `train ≈ val` throughout — *data-bound*, no overfitting.
- The WSD decay tail does disproportionate work: the final 5k iters of decay
  contributed roughly half the total descent.
- Inference quality at perp 170: coherent fragments + heavy repetition loops
  ("history of the history of the history…"). Recognisable as English, not as
  a usable model.

**What this told us to do next.** Scale data first — the cheap, "data-bound" lever.

---

## Run 2 — 30M · 500M tokens · 30k iters

**Setup.** WSD warmup 1000 / decay 5000. ~10 hrs on the Jetson. W&B logging
added (env-var override `NANODIFF_WANDB=1`).

**Result.** Best val **4.72**, perplexity **112**.
**−34% perplexity vs Run 1**, for 5× more data.

**What we learned.**
- Data-bound diagnosis was correct. Simple lever, big win.
- Same three-phase shape (fast cliff → stable-LR plateau → clean decay tail).
- Chinchilla's ~20-tokens-per-param rule (≈ 600M for 30M) puts this run *right
  at* the predicted compute-optimal point — and the model still had room to
  absorb. The rule isn't pessimistic for masked-diffusion at this scale.
- Inference: noticeably longer coherent fragments. Loops still common.

**What this told us to do next.** Push data further to find where the 30M
actually saturates.

---

## Run 3 — 30M · 2B tokens · 125k iters

**Setup.** WSD warmup 4000 / decay 20000. 125k iters = one full epoch over 2B
tokens. ~41 hrs on the Jetson.

**Result.** `ckpt_final.pt` val **4.46** (`eval.py`, 500 batches), perplexity
**86.5**. **−23% perplexity vs Run 2**, for 4× more data.

**What we learned.**

*Scaling dynamics:*
- **Diminishing returns are now visible.** Run 2 bought −34% perp per 5× data;
  Run 3 bought −23% per 4×. The 30M is approaching its capacity floor for
  FineWeb-Edu text. The cheap-data-first lever has mostly paid out.
- **The stable-LR plateau is mostly compute-spent-for-nothing.** Iters
  5000–105000 (78% of the run) moved val from ~5.2 to ~4.85. The last 16%
  (decay) dropped val from ~4.87 to ~4.46. For models near their loss floor,
  *the budget should be weighted toward the decay tail*.

*Eval-noise gotcha (operationally important):*
- The training loop's default `eval_iters=100` is noisy enough that the
  "best val" heuristic *mislabelled the run*. Iter 120000 got a lucky 4.50
  reading and was saved as `ckpt.pt`. Offline eval (500 batches) shows iter
  120000 is actually ~4.58 and iter 125000 is ~4.46. **`ckpt_final.pt` ended
  up meaningfully better than `ckpt.pt`.**
  - Practical rule: **for WSD-decay runs, default to `ckpt_final.pt`** — the
    monotonic LR decay means the final state is typically the best state,
    even if no eval was run on it. Or bump `eval_iters` for accurate
    in-loop ranking.

**What this told us to do next.** Diminishing returns on data are clear → the
question switches to **capacity**. The 50M model on this same 2B-token dataset
is the next experiment.

---

## Final scoreboard so far

| Run | Tokens | Iters | Val (measured properly) | Perplexity | Δ vs prev |
|---|---|---|---|---|---|
| 1 | 100 M | 10 k | 5.14 | 170 | — |
| 2 | 500 M | 30 k | 4.72 | 112 | **−34%** |
| 3 | 2 B | 125 k | **4.46** | **86.5** | **−23%** |

For reference: GPT-2 small (124M params, ~10B tokens) reaches perplexity
~25–30 on standard LM benchmarks. **Closing that gap on a Jetson isn't the
goal** — understanding the scaling regime well enough to scale rationally on
bigger hardware is.

---

## Cross-cutting lessons

1. **The decay tail does most of the work.** Across all three runs, the
   final 15-20% of iters consistently produced the largest single drop in
   val. The stable-LR plateau is mostly wasted compute once the model is
   near its loss floor at that LR. Future runs should weight the budget
   accordingly.

2. **`train ≈ val` everywhere — overfitting never showed up.** Means we
   stayed firmly in the *data-bound regime* across all three runs at 30M.
   Would need a much bigger model or much less data to surface the opposite
   signal.

3. **Diminishing returns are real even at toy scale.** Per-4×-data perplexity
   improvement went `−34% → −23%` across our two scaling steps. Each
   multiplicative data jump buys roughly the same *absolute* nat improvement
   until you near the model's capacity floor — and then noticeably less.

4. **In-run "best val" tracking is only as accurate as `eval_iters`.**
   Default 100 batches is fine for a noisy *signal*, but not accurate enough
   to *rank* nearby checkpoints. For runs you care about, increase it, or
   trust `ckpt_final.pt` for post-WSD-decay runs.

5. **Jetson is great for *learning*, painful for *scaling*.** A 30M run
   takes 3-41 hrs here, and each multiplicative scaling step multiplies that.
   Real capacity sweeps (50M / 150M / 1B) want the DGX-Spark.

---

## Planned next: capacity sweep on DGX-Spark

Run 3 left a clean open question: *the 30M is at its data floor — how much
does adding capacity buy on the same 2B-token dataset?* The answer is a
three-run sweep on the Spark, varying only model size:

| Run | Model | Non-emb params | Tokens | Iters | Effective batch | Est. wall-clock |
|---|---|---|---|---|---|---|
| 4 | 50M  | 49.6 M | 2.1 B | 16 k | 128 seqs | ~2 hrs |
| 5 | 150M | 151.8 M | 2.1 B | 16 k | 128 seqs | ~3 hrs |
| 6 | 350M | 303.6 M | 2.1 B | 16 k | 128 seqs | ~5-6 hrs |

Everything *except model size* is held constant — same data, same iters, same
batch, same LR (sqrt-scaled from the Jetson runs: `6e-4 → 1.2e-3` for the
4× larger effective batch), same WSD schedule proportions. The result is a
**capacity curve at fixed data**, directly comparable to the 30M Run 3 point.

Configs: `configs/train_50m_spark.py`, `train_150m_spark.py`, `train_350m_spark.py`.
Launch: `bash scripts/spark_sweep.sh` (runs all three sequentially with W&B
logging on).

Three outcomes we'd interpret:
- **Val drops sharply with each step up (e.g., 50M ≈ 4.3, 150M ≈ 4.0, 350M ≈ 3.7):**
  the 2B-token dataset has plenty of *information* the 30M just couldn't extract.
  Scale model on this same data is the cheap next move.
- **Val drops then plateaus (e.g., 50M ≈ 4.3, 150M ≈ 4.1, 350M ≈ 4.05):**
  we've hit the *data's* signal limit at this scale — more capacity is now
  data-bound. Scale data next, on a bigger box.
- **All three land near 4.4–4.5:** something else is the bottleneck (likely
  the masked-diffusion objective at this scale, or block_size, or LR). Worth
  investigating before more compute.

The wall-clock budget (~10-12 hrs total) leaves headroom in a 24-40 hr window
for either a 4th larger run (e.g., 700M) or repeating the most informative
point with a different LR / longer schedule.

---

## Portability fixes that came out of these runs

Captured here as a warning for anyone porting nanoDiff to non-standard
hardware. All are in git; this is the "why" beside the "what":

- **`torch.distributed` may not exist in the build.** The Jetson torch wheel
  ships without it. `train.py` imports `DDP`/process-group helpers lazily
  inside the `if ddp:` branch, so single-GPU training works on
  distributed-free builds.
- **`torch.compile` can fail at runtime.** Jetson's `triton` is incompatible
  with this torch's inductor backend. `train.py` sets
  `torch._dynamo.config.suppress_errors = True` before `torch.compile(model)`,
  so a failed compile silently falls back to eager.
- **The sampler could emit the `[MASK]` token as a "generation"**, leaving
  leftover masks in the output (and wasting schedule slots since each commit
  is then a no-op). Fixed by setting the mask-token logit to `-inf` before
  decoding.
- **`steps % n_blocks == 0` was an over-strict assertion** in the sampler.
  Now any `(steps, gen_length, block_length)` triple works as long as
  `steps ≥ n_blocks`; extra steps distribute to the first few blocks.
- **`prepare_data.py` exits via `os._exit(0)`** to skip the `datasets`
  streaming reader's noisy interpreter-shutdown crash. The `.bin` files are
  fully flushed before the hard exit, so no data is lost.
