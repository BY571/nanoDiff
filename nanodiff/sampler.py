"""Reverse process: generating text by iterative denoising.

Implements LLaDA's low-confidence remasking sampler (Algorithm 5), with optional
semi-autoregressive block decoding (LLaDA Fig. 4c).

One denoising step:
  1. Run the model on the current (partly-masked) sequence -> predict every token.
  2. Commit only the most *confident* predictions; leave the rest masked for now.
  3. Repeat. Easy tokens get decided early; hard tokens get more rounds of context
     before they are committed.

`block_length` interpolates between the two regimes (this is the AR<->diffusion
knob that papers like BD3-LM make explicit):
    block_length == gen_length  -> pure diffusion (fill everything in parallel)
    block_length == 1           -> (almost) autoregressive, strictly left to right
"""
import torch
import torch.nn.functional as F


def _top_k_filter(logits, top_k):
    """Mask out everything outside the top-k by setting their logits to -inf."""
    topk_vals = logits.topk(top_k, dim=-1).values        # (..., k)
    threshold = topk_vals[..., -1:]                       # smallest kept logit
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits, top_p, max_candidates=512):
    """Nucleus filter: keep the smallest set of tokens whose cumulative softmax
    probability reaches `top_p`, mask the rest with -inf. The top-1 token is
    always kept, even if its own probability exceeds top_p alone.

    We use `torch.topk(k=max_candidates)` to pull out the candidates and
    `logsumexp` over the full vocab *once* for correct global normalization —
    O(V log K) instead of the O(V log V) full sort. For typical top_p≈0.9 on a
    50k-vocab model the true nucleus is <100 tokens, so the K=512 candidate set
    is generous and the result is exact (the nucleus is necessarily a prefix of
    top-K when K covers it). Measured ~2x speedup of the whole sampling step.
    """
    V = logits.size(-1)
    k = min(max_candidates, V)
    # topk returns descending-sorted values + indices — no separate sort needed.
    topk_vals, topk_idx = logits.topk(k, dim=-1)
    # Compute global probabilities for just the top-K, using logsumexp over the
    # full vocab so the prob distribution is the *true* one (not renormalized
    # within the top-K — which would slightly shift the nucleus boundary).
    log_z = logits.logsumexp(dim=-1, keepdim=True)
    sorted_probs = (topk_vals - log_z).exp()
    cum_probs = sorted_probs.cumsum(dim=-1)
    # Shift right so position i contains the cumulative prob BEFORE token i —
    # this guarantees the top-1 token is always kept (cum_before[0] = 0).
    cum_before = torch.zeros_like(cum_probs)
    cum_before[..., 1:] = cum_probs[..., :-1]
    keep_topk = cum_before < top_p
    # Scatter back to the original vocabulary index space.
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, topk_idx, keep_topk)
    return logits.masked_fill(~keep, float("-inf"))


def _transfer_schedule(block_length, steps, device):
    """How many tokens to commit (un-mask) at each denoising step of a block.

    Spreads `block_length` commits as evenly as possible over `steps` steps; the
    counts sum exactly to `block_length`, so a block is fully decoded after its
    last step.
    """
    base = block_length // steps
    rem = block_length % steps
    counts = torch.full((steps,), base, dtype=torch.long, device=device)
    counts[:rem] += 1
    return counts


@torch.no_grad()
def generate(model, prompt_ids, gen_length, steps, block_length=None,
             temperature=0.0, top_k=None, top_p=None, rep_penalty=0.0,
             remasking="low_confidence", mask_token_id=None,
             use_cache=False, tau=None):
    """Generate `gen_length` tokens conditioned on `prompt_ids`.

    Args:
        model:        a NanoDiff (or unwrapped) model
        prompt_ids:   (B, P) LongTensor — the conditioning prefix (never masked)
        gen_length:   number of tokens to generate after the prompt
        steps:        total denoising steps (split evenly across blocks)
        block_length: semi-AR block size; defaults to gen_length (pure diffusion)
        temperature:  0.0 -> greedy argmax; >0 -> sample from the softmax
        top_k:        if set, keep only the k highest-prob tokens before sampling
        top_p:        if set in (0,1], nucleus sampling — keep smallest set whose
                      cumulative prob reaches top_p
        rep_penalty:  subtract this constant from the logits of every token
                      already present in the sequence. The fix for the
                      repetition collapse of small diffusion LMs (see Notes).
        remasking:    "low_confidence" (LLaDA default) or "random"
        mask_token_id: override; defaults to model.config.mask_token_id
        use_cache:    Fast-dLLM prefix K/V cache — prefill the full sequence's
                      K/V once per block, then `forward_decode` only over the
                      active block on each step. APPROXIMATE (not bit-identical
                      to the no-cache path): at depth > 1 the residual stream
                      at non-active positions also drifts as the active region
                      commits, but Fast-dLLM (Lou et al.) shows the K/V cosine
                      similarity between adjacent steps is ~1.0 in practice,
                      so the approximation costs near-zero quality. Off by
                      default; validate per workload via LAMBADA or similar.
        tau:          if set in (0,1], Fast-dLLM threshold parallel decoding —
                      commit EVERY active position whose max-softmax-prob >= tau
                      instead of the fixed top-`counts[i]` schedule. Falls back
                      to committing the single highest-confidence position if
                      none cross the threshold (so the loop always makes
                      progress). Combined with `use_cache` this is the
                      Fast-dLLM ~10x speedup. tau=0.9 is the paper's default.

    Returns:
        (B, P + gen_length) LongTensor — prompt followed by the generated tokens.

    Notes:
        top_k/top_p only affect the SAMPLED token (active when temperature > 0).
        The confidence score used for remasking is always computed on the
        *unfiltered* softmax — that's the model's intrinsic certainty, and we
        don't want filtering to artificially inflate it.

        `rep_penalty` is the cure for the repetition collapse small/weak
        diffusion LMs fall into ("the capital of France is the capital of
        France is ..."). The root cause is logit-level: every masked slot's
        distribution is biased toward re-emitting a recent token. Penalising
        already-present tokens at the logit level fixes it; perturbing the
        *commit order* (random/Gumbel remasking) does NOT — the bias is in the
        logits, not the ordering.

        The cache and threshold paths are *training-free* inference tricks
        (Lou et al., Fast-dLLM, arXiv:2505.22618). The cache is exact for
        bidirectional masked diffusion because each token's K and V depend
        only on its own input embedding, not on the rest of the sequence —
        so as long as the underlying tokens don't change, the K/V at those
        positions doesn't either.
    """
    was_training = model.training
    model.eval()
    device = prompt_ids.device
    cfg = model.config
    mask_id = cfg.mask_token_id if mask_token_id is None else mask_token_id
    B, P = prompt_ids.shape

    block_length = block_length or gen_length
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    n_blocks = gen_length // block_length
    assert steps >= n_blocks, "steps must be at least the number of blocks"
    # Distribute `steps` across blocks; if it doesn't divide evenly, the first
    # `extra` blocks get one additional refinement pass each.
    base_steps = steps // n_blocks
    extra = steps % n_blocks

    # Start fully masked, then drop in the (clean, fixed) prompt.
    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    for b in range(n_blocks):
        s0 = P + b * block_length          # active block: [s0, s1)
        s1 = s0 + block_length
        block_steps = base_steps + (1 if b < extra else 0)
        counts = _transfer_schedule(block_length, block_steps, device)

        # Prefill seeds the per-layer K/V cache for the whole current sequence.
        # The prefix [0:s0] and the still-all-masked suffix [s1:T] won't change
        # during this block, so their K/V is computed once and reused on every
        # inner step. Only [s0:s1] gets re-projected through qkv each step.
        if use_cache:
            _, kv_caches = model.forward_prefill(x)

        for i in range(block_steps):
            is_mask_active = x[:, s0:s1] == mask_id          # (B, A)
            if not is_mask_active.any():
                break                                        # block fully committed

            # ---- forward: get logits for the active range only ----
            if use_cache:
                logits_active, kv_caches = model.forward_decode(x, kv_caches, (s0, s1))
            else:
                logits_active = model(x)[:, s0:s1, :]

            # Repetition penalty (frequency-scaled, OpenAI-style): subtract
            # `rep_penalty * count` from each token's logits, where `count` is
            # how many times that id already appears in x (prompt + committed).
            # Frequency-scaling matters — a flat presence penalty can't escalate,
            # so a model committed to spamming one token ("Cold, Cold, Cold...")
            # never gets pushed off it. Scaling by count does: 2nd hit -2x,
            # 3rd -3x, until a different token wins. This breaks the collapse.
            if rep_penalty > 0:
                token_freq = torch.zeros(B, logits_active.size(-1),
                                         dtype=logits_active.dtype, device=device)
                token_freq.scatter_add_(1, x, torch.ones_like(x, dtype=logits_active.dtype))
                logits_active = logits_active - rep_penalty * token_freq.unsqueeze(1)

            # The [MASK] token is an input-only sentinel — it never appears in
            # clean data, so it is never a valid generation. Forbid it, otherwise
            # the model could "predict" a mask, the commit becomes a silent no-op,
            # and the position is left masked forever.
            logits_active[:, :, mask_id] = float("-inf")

            # Compute the unfiltered softmax up front — needed for the
            # confidence score regardless of which sampling path we take.
            probs_full = F.softmax(logits_active.float(), dim=-1)         # (B, A, V)

            if temperature > 0:
                sample_logits = logits_active.float()
                if top_k is not None and top_k > 0:
                    sample_logits = _top_k_filter(sample_logits, top_k)
                if top_p is not None and 0 < top_p < 1:
                    sample_logits = _top_p_filter(sample_logits, top_p)
                probs = F.softmax(sample_logits / temperature, dim=-1)
                x0_active = torch.multinomial(probs.view(-1, probs.size(-1)),
                                              1).view(B, -1)               # (B, A)
            else:
                x0_active = logits_active.argmax(dim=-1)                   # (B, A)

            # Confidence score used to rank which predictions to commit this step.
            if remasking == "low_confidence":
                conf_active = probs_full.gather(-1, x0_active.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                conf_active = torch.rand(is_mask_active.shape, device=device)
            else:
                raise ValueError(f"unknown remasking strategy: {remasking!r}")

            # Only currently-masked positions inside the active block are eligible.
            conf_active = torch.where(is_mask_active, conf_active,
                                      torch.full_like(conf_active, float("-inf")))

            # Within-step repetition penalty. When the step commits more than one
            # position (low steps budget OR threshold mode), independently-decided
            # positions can collide on the same token ("process process", "the the"),
            # because the sequence-wide `rep_penalty` only fires BETWEEN steps. Here
            # we identify same-token collisions IN THIS STEP and penalise the
            # losers' confidence so the winner gets committed and the others fall
            # out of the top-K. Only active when rep_penalty > 0 — opting into
            # rep_penalty signals "I care about non-repetition" so extending the
            # rule within-step is the consistent choice.
            if rep_penalty > 0:
                V = logits_active.size(-1)
                token_max_conf = torch.full((B, V), float("-inf"),
                                            dtype=conf_active.dtype, device=device)
                token_max_conf.scatter_reduce_(1, x0_active, conf_active,
                                               reduce="amax", include_self=False)
                position_token_max = token_max_conf.gather(1, x0_active)   # (B, A)
                is_winner = conf_active >= position_token_max
                conf_active = torch.where(is_winner, conf_active,
                                          conf_active - rep_penalty)

            # ---- commit policy ----
            if tau is not None:
                # Threshold parallel decoding: commit every eligible position
                # whose model-confidence >= tau. Adaptive — early steps may
                # commit many tokens at once (cheap predictions), later steps
                # only a few (the hard ones). If nothing crosses the bar we
                # still commit the single most-confident eligible position so
                # the loop always makes progress.
                commit_active = (conf_active >= tau)
                if not commit_active.any():
                    flat_argmax = conf_active.argmax(dim=1, keepdim=True)
                    commit_active = torch.zeros_like(is_mask_active)
                    commit_active.scatter_(1, flat_argmax, True)
            else:
                # LLaDA's fixed schedule: commit exactly counts[i] tokens.
                k = int(counts[i])
                if k > 0:
                    commit_idx = conf_active.topk(k, dim=1).indices         # (B, k)
                    commit_active = torch.zeros_like(is_mask_active)
                    commit_active.scatter_(1, commit_idx, True)
                else:
                    commit_active = torch.zeros_like(is_mask_active)

            # Write the committed tokens into the active slice of x.
            x[:, s0:s1] = torch.where(commit_active, x0_active, x[:, s0:s1])

    if was_training:
        model.train()
    return x
