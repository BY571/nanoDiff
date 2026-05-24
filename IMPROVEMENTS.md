# Improvements & Provenance

A running ledger of everything that changed nanoDiff's *speed* or *architecture*,
with the source it came from (paper, blog post, our own measurement) and the
commit that landed it. The point is the **tree of provenance**: a future reader
should be able to look at any non-obvious optimization in the codebase and find
the paper or experiment that justified it.

> **Add an entry whenever** you ship a speed / architecture change driven by
> external research. **Also add an entry** whenever you investigate something
> but decide not to ship it (with a one-line rationale) — that prevents the
> next person from re-investigating the same idea.

---

## Inference

### Landed

| Commit | Change | Source | Impact | Notes |
|---|---|---|---|---|
| [`bd96911`](../../commit/bd96911) | `_top_p_filter` rewritten as `topk + logsumexp` (no full vocab sort) | Our own measurement (full-vocab sort was ~50% of step time at chat defaults) | **1.72×** on the top-p path | Exact (same nucleus) for top_p ≤ ~0.95 |
| [`bd96911`](../../commit/bd96911) | SDPA backend lock — whitelist FLASH + MATH | Our own measurement: EFFICIENT and CUDNN both produce >1% drift vs FLASH on sm_121 (1.4e-2 rel diff measured). Backs up [NVIDIA dev-forum report](https://forums.developer.nvidia.com/t/dgx-spark-sm121-silent-sdpa-efficient-attention-corruption-in-a-custom-pytorch-build-diagnostic-chain-standalone-reproducer-workaround/368005) | correctness shield | PyTorch already picked FLASH; this prevents future drift |
| [`bd96911`](../../commit/bd96911) | chat.py honest tok/s metric (raw `gen_length/dt`, not post-EOT-truncation) | Our own audit | **~3.5× displayed** (purely reporting, no real speedup) | Lesson: audit metrics before optimizing |
| [`fa29047`](../../commit/fa29047) | Fast-dLLM block-wise prefix K/V cache (`--use-cache`) | [Fast-dLLM v1 — Lou et al. 2025](https://arxiv.org/abs/2505.22618) ([code](https://github.com/NVlabs/Fast-dLLM)) | 1.0-1.16× alone (size-dependent) | Approximate (deeper-layer K/V drifts); LAMBADA-equivalent 15.74→15.72% |
| [`fa29047`](../../commit/fa29047) | Confidence-threshold parallel decoding (`--tau`) | [Fast-dLLM v1 §3.3](https://arxiv.org/abs/2505.22618) | 1.45-1.50× | Stacks with `--use-cache`; conflicts with `--compile` |
| [`fa29047`](../../commit/fa29047) | Sampler refactor: rep_penalty/softmax/conf operate on `(B, A, V)` active slice instead of full `(B, T, V)` | Our own (fell out of the cache rewrite) | **1.45×** in chat-default workload | Side effect that ended up bigger than the cache itself |
| [`a051c7f`](../../commit/a051c7f) | Within-step rep_penalty (penalize same-token collisions inside one denoising step) | Our own, motivated by observed "process process" doubling at `--steps 32` | enables `--steps 32` without doubling artifacts | Used `scatter_reduce(amax)` to find each token's highest-conf-predicting position |
| [`04651c7`](../../commit/04651c7) | `torch.compile` re-enabled for chat (`--compile`) | PyTorch native | **1.36×** alone, **4.4×** combined with `--steps 32` | First-call warmup ~5-30s. Conflicts with `--use-cache` (same overhead target). The earlier "compile slows generation" finding was an artifact of the pre-refactor sampler |
| [`04651c7`](../../commit/04651c7) | `--steps 32` recommendation (from default 96) | Our own bench + the within-step rep_penalty fix above | **2.78×** alone | Safe because of the within-step fix; below ~24 some across-step doubling reappears |

**Current best**: `--compile --steps 32` → **1034 tok/s** on 150M SFT (DGX Spark, chat defaults). **4.4× over baseline**, ~13× over original displayed tok/s.

### Proposed / Investigated

| Priority | Item | Source | Expected impact | Cost | Rationale |
|---|---|---|---|---|---|
| **HIGH** | Prophet early-commit (top1-top2 confidence gap → commit entire remainder) | [Prophet arXiv 2508.19982](https://arxiv.org/abs/2508.19982) | 1.15-1.40× | low (~30 lines in `sampler.py`) | Orthogonal to `--tau`: tau commits *positions* above threshold, Prophet commits *entire remainder* when uniformly confident |
| MEDIUM | `torch.compile(mode="reduce-overhead")` (CUDA-graph capture) | [PyTorch docs](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) | 1.05-1.15× over `mode="default"` | trivial (one string) | Need static shapes; our sampler is a good fit since the Phase B refactor |
| MEDIUM | INT8 weight-only quantization | [torchao](https://github.com/pytorch/ao) | 1.1-1.3× at batch=1 | low (one-line `quantize_`) | Quality cost on 150M is non-trivial; measure perplexity first. Composes with `torch.compile` |
| MEDIUM | Fast-dLLM v2 DualCache (suffix cache) | [Fast-dLLM v2 arXiv 2509.26328](https://arxiv.org/abs/2509.26328) ([page](https://nvlabs.github.io/Fast-dLLM/v2/)) | 1.10-1.25× over v1 | medium | **Requires block-causal training**; our LLaDA-style full-bidirectional model won't get the full win without retraining |
| LOW | dLLM-Cache (interval-refresh + V-verify) | [arXiv 2506.06295](https://arxiv.org/abs/2506.06295) ([code](https://github.com/maomaocun/dLLM-cache)) | uncertain at 150M (paper tests 8B+) | medium | Conceptually disjoint from Fast-dLLM v1; would replace, not stack |
| WAIT | Spiffy / SSD / DiffuSpec — speculative decoding for dLLMs | [Spiffy arXiv 2509.18085](https://arxiv.org/abs/2509.18085) · [DiffuSpec arXiv 2510.02358](https://arxiv.org/abs/2510.02358) · [SSD arXiv 2510.04147](https://arxiv.org/abs/2510.04147) | 1.3-1.7× over our (cache + τ) baseline | high | **No public code yet** (Q1-Q2 2026 expected). Tree-verification is intricate to implement from paper |
| SKIP | Sparse-dLLM / SparseD attention sparsity | [Sparse-dLLM arXiv 2508.02558](https://arxiv.org/abs/2508.02558) · [SparseD arXiv 2509.24014](https://arxiv.org/abs/2509.24014) | negligible at gen=96, steps=32 | high | Sparsity wins only at long context + many steps |
| SKIP | cuDNN-SDPA backend (LLaDA 2.0's choice) | [LLaDA 2.0 arXiv 2512.15745](https://arxiv.org/abs/2512.15745) | would be 1.3× | n/a | **Broken on sm_121** — measured 1.4% rel drift vs FLASH |
| SKIP | FlashAttention-3 install | community wheels | would be marginal | high | FA3 will not compile on sm_121; PyTorch SDPA's FLASH path is the right backend on our hardware |

---

## Training

### Landed (initial nanoDiff setup)

| Change | Source | Notes |
|---|---|---|
| `torch.compile(mode="default")` | PyTorch native | ~1.4-1.7× over eager. Falls back to eager on Triton-less builds |
| bf16 autocast | PyTorch native | 2× vs fp32; fits 350M training on a single GB10 |
| Fused AdamW (`fused=True`) | PyTorch native | Marginal but free; on by default in `configure_optimizers` |
| TF32 enabled (`allow_tf32 = True`) | PyTorch native | Marginal |
| `F.cross_entropy` on bf16 logits (no `.float()` cast) | Our own (`nanodiff/diffusion.py:55-60`) | Saves ~52 GB of activation memory at B=128, T=1024, V=50304 — critical for fitting big batches |
| `F.scaled_dot_product_attention(is_causal=False)` (PyTorch SDPA → FlashAttention path) | PyTorch native, validated against [sm_121 corruption report](https://forums.developer.nvidia.com/t/dgx-spark-sm121-silent-sdpa-efficient-attention-corruption-in-a-custom-pytorch-build-diagnostic-chain-standalone-reproducer-workaround/368005) | bidirectional attention via PyTorch's FlashAttention. Locked to FLASH on sm_121 in [`bd96911`](../../commit/bd96911) |

### Proposed / Investigated

| Priority | Item | Source | Expected impact | Cost | Rationale |
|---|---|---|---|---|---|
| MEDIUM | Liger Kernel: RMSNorm + RoPE + SwiGLU drop-ins | [linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) | 3-5% step time | low | Triton-based, should work on sm_121. Drop-in replacements |
| MEDIUM | Custom Triton chunked-CE with per-token 1/t weight | Inspired by [Liger FLCE](https://github.com/linkedin/Liger-Kernel) | 5-10% step time | medium-high | **Liger's FLCE does NOT support per-token weighting** — it has per-class `ce_weight` but not per-row scaling. Our 1/t weighting blocks the drop-in. Need a custom kernel modeled on theirs |
| LOW | `torch.compile(mode="max-autotune-no-cudagraphs")` for training | PyTorch | <5%, risky on sm_121 | low | The `max_autotune_gemm` path is gated by sm-count check on sm_121 (["Not enough SMs"](https://discuss.pytorch.org/t/torch-compile-warning-not-enough-sms-to-use-max-autotune-gemm-mode/184405)) |
| LOW | Cache `TORCHINDUCTOR_CACHE_DIR` across runs | PyTorch | first-call warmup drops to ~0.5s | trivial | Useful for shipped artifacts / dev iteration |
| SKIP | Activation checkpointing | — | net negative at our scale | low | We're not OOM at batch=64; checkpointing trades ~30% compute for 2× batch, not worth at our regime |
| SKIP | Lion / StableAdamW optimizers | various | same speed, mixed quality | low | No wall-clock win, fused AdamW is fine |
| WAIT | fp8 training | [PyTorch fp8 docs](https://docs.pytorch.org/torchao/main/index.html) | premature | high | sm_121 fp8 path overlaps with the `max_autotune_gemm` gate |
| WAIT | GQA (grouped-query attention) | [LLaDA 2.0 §4](https://arxiv.org/abs/2512.15745) | 10-15% KV cache savings | requires retrain | Worth doing on the next architecture-revision train |

---

## How to add an entry

When you implement a speed / architecture change driven by external research:

1. Add a row to **Landed** in the right section.
2. Cite the source as a markdown link — paper URL, blog post, our own measurement, etc.
3. Include the commit hash as a clickable link.
4. Record the **measured** impact (or `qualitative` if you didn't measure).
5. Note what it composes with / conflicts with.

When you investigate something and decline to ship it:

- Add it to **Proposed / Investigated** with priority `SKIP` or `WAIT` and a one-line reason.

Anchor terms used in this doc:

- **Composes with X**: stacks multiplicatively with optimization X.
- **Conflicts with X**: targets the same overhead as X; one cancels the other.
- **Approximate**: not bit-identical to the unoptimized version, but quality measured equivalent on a defensible benchmark.

---

## Related artifacts

- `README.md` → `## Sampling speed` — user-facing summary of what to actually run
- `EXPERIMENTS.md` (gitignored, on-laptop only) — detailed lab notebook for individual investigations
- `nanodiff/sampler.py` / `nanodiff/model.py` — citations inline in docstrings
- `benchmark/README.md` — what we evaluate against
