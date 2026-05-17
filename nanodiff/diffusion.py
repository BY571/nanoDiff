"""The diffusion process: the forward (corruption) process and the training loss.

This is the entire "diffusion" in a masked diffusion LM, and it is tiny.

Forward process (corruption) — LLaDA / MDLM:
    sample t ~ U(t_eps, 1) once per sequence;
    replace each token independently with [MASK] with probability t.
At t -> 0 the sequence is clean; at t -> 1 it is fully masked.

Training loss — LLaDA Algorithm 1 / Eq. 12:
    L = E_{t, x0, x_t} [ -(1/t) * (1/T) * sum_i  1[i masked] * log p_theta(x0_i | x_t) ]
i.e. a cross-entropy on the masked positions only, importance-weighted by 1/t.
The 1/t factor corrects for the fact that lightly-masked sequences produce fewer
training signals; with it, L is a provable upper bound on the negative
log-likelihood (MDLM shows the masked-diffusion ELBO collapses to exactly this).
"""
import torch
import torch.nn.functional as F


def forward_process(x0, mask_token_id, t_eps=1e-3, t=None):
    """Corrupt a clean batch x0 (B, T) into x_t by absorbing-state masking.

    Args:
        x0:            (B, T) clean token ids
        mask_token_id: id of the [MASK] / absorbing-state token
        t_eps:         lower clamp for t so the 1/t loss weight stays finite
        t:             optional (B,) pre-sampled mask probabilities; else U(t_eps, 1)

    Returns:
        x_t:  (B, T) corrupted tokens ([MASK] at masked positions)
        mask: (B, T) bool, True where a token was masked
        t:    (B,)   the per-sequence mask probability actually used
    """
    B, T = x0.shape
    if t is None:
        t = torch.rand(B, device=x0.device) * (1.0 - t_eps) + t_eps
    mask = torch.rand(B, T, device=x0.device) < t.unsqueeze(1)
    x_t = torch.where(mask, mask_token_id, x0)
    return x_t, mask, t


def diffusion_loss(logits, x0, mask, t):
    """LLaDA training objective.

    Args:
        logits: (B, T, V) model predictions for x_t
        x0:     (B, T)    the clean targets
        mask:   (B, T)    bool, True where the token was masked (loss applies here)
        t:      (B,)      per-sequence mask probability
    Returns:
        scalar loss (a stochastic upper bound on the negative log-likelihood).
    """
    B, T, V = logits.shape
    # NOTE: we pass bf16 logits straight to cross_entropy — `F.cross_entropy`
    # already upcasts to fp32 *internally* for log_softmax (numerical stability)
    # but does not store an fp32 copy in the autograd graph. Skipping `.float()`
    # here saves ~B*T*V*4 bytes for the cast + ~same for its gradient — at
    # B=128, T=1024, V=50304 that's ~52 GB. Critical for fitting big-batch
    # training in a single-GPU box's memory.
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        x0.reshape(-1),
        reduction="none",
    ).reshape(B, T)
    ce = ce * mask                       # keep the loss on masked positions only
    per_seq = ce.sum(dim=1) / (t * T)    # 1/t importance weight, normalized by length
    return per_seq.mean()
