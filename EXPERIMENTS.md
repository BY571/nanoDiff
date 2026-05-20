# Experiment Log

Empirical findings from each training run, in chronological order. What matters
here is the *dynamics* and what they imply about the next experiment, not the
absolute numbers (those date instantly).

The log has two phases:
- **Runs 1-3 (Jetson phase)** — a 30M-param model (`configs/train_30m*.py`):
  6 layers, n_embd 384, n_head 6, block_size 512, GPT-2 BPE, masked-diffusion
  objective, FineWeb-Edu (`sample-10BT`). Trained on a Jetson Orin (eager mode,
  bf16 autocast, no compile, no DDP — the Jetson torch build supports none).
- **Runs 4+ (Spark phase)** — capacity and schedule sweeps on the DGX-Spark
  (GB10, bf16, `torch.compile`), scaling the model and then probing the LR
  schedule. Same masked-diffusion objective and FineWeb-Edu data throughout.

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

6. **The GPU saturates — batch size is not a speed knob.** On the Spark,
   throughput is a flat ~37-41K tok/s across every `(block_size, batch)`
   combination. Bigger batch just packs more work per step at proportionally
   longer step time. The real speed knob is *total work* — i.e. `block_size`.

7. **Capacity without a schedule fix buys nothing.** The 150M tied the 50M
   (3.94 vs 3.92) because both ran the same too-long-stable WSD schedule.
   Spending parameters before spending thought on the schedule is wasted
   compute. Fix the schedule, *then* test capacity.

8. **The LR schedule shape is a near-flat surface — with one cliff.** Clean
   eval: WSD-short-decay (4.199) ≈ WSD-long-decay (4.204) — the decay fraction
   (19% vs 62%) is within noise. But cosine (4.248), with no stable phase at
   all, is ~0.05 nats worse — a real gap. Keep *a* stable phase; its length
   and the decay length don't matter. We spent ~15 Spark-hours to learn the
   knob barely moves — a useful negative result, but a negative result.

9. **The repetition collapse was the sampler — fixed by a frequency penalty.**
   The 50M base (val 3.92, PPL 50) decoded to pure loops ("the capital of
   France is the capital of France is…"). A controlled sweep showed why: it's
   a *logit-level* bias — every masked slot's distribution favours re-emitting
   a recent token. Commit-order changes (random / Gumbel-noised remasking) do
   NOT fix it; the bias is in the logits, not the ordering. A frequency-scaled
   repetition penalty (subtract `rep_penalty × token_count` from the logits)
   does — the same checkpoint then produces varied, fluent English. LLaDA's
   low-confidence remasking is fine to keep. The deeper lesson: **the LLaDA
   sampler defaults assume a model strong enough that confidence ≈ correctness;
   small models additionally need the repetition penalty.**

10. **Sampler fixed ≠ model good.** With the repetition penalty the 50M base is
    fluent at the sentence level but confabulates facts ("founded by Louis XIV
    in 1515") and turns incoherent on harder prompts. That residual is the
    genuine 50M / 2B-token limitation — and the real target for SFT and scale.
    The clean split: loops were the *sampler*; confabulation is the *model*.

11. **SFT teaches format, fast — not knowledge.** Fine-tuning the base on
    Alpaca-cleaned drove the loss 7.77 → ~1.43 within 200 iters, then flat for
    the remaining 2200. Instruction-following is a low-entropy behavior and
    transfers almost instantly. It does not lift the factual ceiling — the SFT
    model answers on-topic but still confabulates. Budget SFT runs short.

---

## Run 4 — 50M · 2B tokens · 16k iters · DGX-Spark

**Setup.** block_size 1024, effective batch 128, WSD warmup 500 / decay 3000,
lr 1.2e-3. ~12 hrs on the Spark.

**Result.** `ckpt_final.pt` val **3.92** (`eval.py`, 500 batches), perplexity
**50.6**.

**What we learned.**
- Big jump vs the 30M Run 3 (4.46 → 3.92), but the comparison is confounded:
  the Spark run sees ~4× more *effective* tokens per iter (batch 128 vs 32) and
  finishes a full pass over the 2B set. At matched tokens-seen it's a smaller
  gain — capacity helped, but less than the raw number suggests.
- Same three-phase WSD shape as every Jetson run.

---

## Run 5 — 150M · 2B tokens · 16k iters · DGX-Spark

**Setup.** Identical to Run 4 except model size (12 layers, n_embd 1024) and
microbatch (64 × grad_accum 2 — 150M OOMs at microbatch 128). ~22 hrs.

**Result.** val **3.94** — **a tie with the 50M.**

**What we learned.**
- **3× the parameters bought nothing.** The 150M landed at the same val as the
  50M. With the schedule, LR and data held identical, the bottleneck is clearly
  *not capacity*.
- Chinchilla framing: 50M wants ~1B tokens (we gave 2B → over-trained), 150M
  wants ~3B (we gave 2B → under-trained). Both converge to ~3.93 because the
  **schedule** dominates, not the model size.
- This killed the planned 350M run — repeating the same setup at a bigger model
  would have burned ~24 hrs to reproduce the tie.

**What this told us to do next.** Stop scaling model size; fix the schedule.

---

## block_size investigation — 1024 → 512

A microbench on the Spark (50M, fwd+bwd, bf16) showed throughput is a flat
~37-41K tokens/sec across every `(block_size, batch)` combination — **the GPU
is saturated; batch size is not a speed knob.** What *is* a knob: halving
`block_size` halves work-per-step.

| block_size | batch | ms/step (compiled) | note |
|---|---|---|---|
| 1024 | 128 | 3640 (eager bench) | original baseline |
| 512  | 128 | 1128 | **~3.2× faster** with compile |

At 50M scale, masked-diffusion learning is local — a 512-token window loses
almost nothing per-token. So block_size 512 became the new default: ~2× faster
wall-clock, and `max_iters` 16k now sees ~1B tokens (Chinchilla-optimal for
50M instead of the over-trained 2B).

---

## Runs 6-8 — 50M v2 schedule sweep · block_size 512 · 1B tokens

Run 5 proved the schedule, not capacity, is the bottleneck. This sweep varies
**only the LR schedule shape** — same 50M model, same block_size 512, same 1B
tokens, same lr 1.2e-3 → 1e-5, same 16k iters. Chained autonomously overnight
by `scripts/auto_chain_sweep.sh`.

| Run | Schedule | Shape | wandb (eval_iters=100) | clean eval (500 batches) |
|---|---|---|---|---|
| 6 | WSD baseline | warmup 500, stable→13k, decay 13k→16k | 4.205 | **4.199** |
| 7 | Cosine | warmup 500, half-cosine 500→16k, no stable | 4.221 | 4.248 |
| 8 | WSD long-decay | warmup 500, stable→6k, decay 6k→16k | 4.188 | 4.204 |

**What we learned.**
- **The noisy in-loop eval lied about the ranking.** wandb's `eval_iters=100`
  had long-decay winning by 0.017; the clean 500-batch eval put WSD-baseline
  ahead by 0.005. Lesson #4 striking again — *always* confirm a sweep with a
  proper offline eval before drawing conclusions.
- **The decay fraction barely matters.** WSD-baseline (19% decay) and
  long-decay (62% decay) landed 0.005 nats apart — pure noise. Whether you
  decay over 3k or 10k iters is irrelevant at this scale.
- **Cosine is genuinely worse** — 0.049 nats behind, a gap *outside* the noise
  band. Removing the stable phase *entirely* hurts. So: keep a stable phase,
  but its length (and the decay length) is not a sensitive knob.
- Net: **the LR schedule shape is a near-flat optimization surface here**, with
  one cliff (don't go full cosine). We spent ~15 Spark-hours to learn the knob
  we were turning barely moves the needle — itself a useful negative result.

> Note: these ~4.2 vals are worse than Run 4's 3.92 because the v2 runs see
> 1B tokens vs Run 4's 2B — the halved-token-budget effect, not a schedule
> regression. The sweep is internally valid (all three saw the same 1B).

---

## Spark scoreboard

| Run | Model | block | Tokens | Schedule | Val | Perplexity |
|---|---|---|---|---|---|---|
| 4 | 50M  | 1024 | 2B | WSD 3k decay | 3.92 | 50.6 |
| 5 | 150M | 1024 | 2B | WSD 3k decay | 3.94 | 51.4 |
| 6 | 50M  | 512  | 1B | WSD 3k decay | 4.199 | 66.6 |
| 7 | 50M  | 512  | 1B | cosine | 4.248 | 70.0 |
| 8 | 50M  | 512  | 1B | WSD 10k decay | 4.204 | 66.9 |

All Spark vals above are clean offline eval (`eval.py`, 500 batches).

---

## Run 9 — 50M SFT on Alpaca-cleaned

The first non-pretraining run: supervised fine-tuning the Run 4 base (val 3.92)
into an instruction-follower, via the LLaDA Algorithm 2 recipe (response-only
masking — see `nanodiff/sft.py`).

**Setup.** Init from `50m_spark/ckpt_final.pt`; dataset yahma/alpaca-cleaned
(50,760 train / 1,000 val); prompt/response 256/256; batch 64; cosine LR
1e-4 → 1e-5, warmup 100; 2400 iters. ~20 min on the Spark.

**Result.** SFT response NLL-bound **~1.41**. The model goes from *continuing*
text to *answering* instructions — on-topic, in the response format, ending
cleanly when `chat.py --sft` truncates at the learned `<|endoftext|>` marker.
Content is 50M-limited: fluent but confabulates ("founded by Louis XIV").

**What we learned.**
- **SFT format is learned almost instantly.** The loss fell 7.77 → ~1.43 by
  iter 200 and then *barely moved* through iter 2400. SFT teaches a low-entropy
  *format*, not knowledge — ~400 iters would have sufficed; 2400 was ~6× over.
- **The structural win is real and complete** — instruction-following works.
  The residual (wrong facts, occasional incoherence) is the base model's
  capacity ceiling, exactly as Run 5 predicted; SFT cannot raise it.
- **EOT-for-length-control works.** Padding responses with `<|endoftext|>` and
  keeping the pads in the loss taught the model to emit an end-marker;
  truncating the response there cleanly removes the junk tail.

Published: [Sebasdi/nanodiff-50m-sft-alpaca](https://huggingface.co/Sebasdi/nanodiff-50m-sft-alpaca).

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
