"""Tests for the SFT forward process and loss (nanodiff/sft.py).

Same standalone-assertion style as smoke_test.py — run directly, exits
non-zero on any failure so it doubles as a CI gate.

    uv run python test_sft.py
"""
import sys

import torch

from nanodiff.sft import sft_forward_process, sft_loss

MASK_ID = 127


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return bool(ok)


def main():
    torch.manual_seed(0)
    B, P, L, V = 4, 6, 8, 128
    # token ids strictly below MASK_ID, so a real token is never confused with [MASK]
    prompt_ids = torch.randint(0, MASK_ID, (B, P))
    response_ids = torch.randint(0, MASK_ID, (B, L))
    x0 = torch.cat([prompt_ids, response_ids], dim=1)          # (B, P+L)

    results = []

    # ---- sft_forward_process ----------------------------------------------
    x_t, resp_mask, t = sft_forward_process(prompt_ids, response_ids, MASK_ID)

    results.append(check(
        "forward: shapes — x_t (B,P+L), resp_mask (B,P+L), t (B,)",
        x_t.shape == (B, P + L) and resp_mask.shape == (B, P + L)
        and t.shape == (B,)))

    results.append(check(
        "forward: prompt span is never corrupted",
        torch.equal(x_t[:, :P], prompt_ids)))

    results.append(check(
        "forward: response_mask is all False over the prompt span",
        not resp_mask[:, :P].any()))

    results.append(check(
        "forward: every masked position holds [MASK]",
        (x_t[resp_mask] == MASK_ID).all().item()))

    results.append(check(
        "forward: every unmasked position holds the original token",
        torch.equal(x_t[~resp_mask], x0[~resp_mask])))

    results.append(check(
        "forward: t in [t_eps, 1]",
        (t >= 1e-3).all().item() and (t <= 1.0).all().item()))

    # t = 1.0 -> the entire response span is masked, prompt still intact
    x_t1, resp_mask1, _ = sft_forward_process(
        prompt_ids, response_ids, MASK_ID, t=torch.ones(B))
    results.append(check(
        "forward: t=1 masks the whole response span, prompt untouched",
        resp_mask1[:, P:].all().item()
        and not resp_mask1[:, :P].any()
        and torch.equal(x_t1[:, :P], prompt_ids)))

    # ---- sft_loss ---------------------------------------------------------
    logits = torch.randn(B, P + L, V, requires_grad=True)
    loss = sft_loss(logits, response_ids, resp_mask, t)
    results.append(check(
        "loss: returns a finite scalar",
        loss.dim() == 0 and torch.isfinite(loss).item()))

    # perfect predictions on the response span -> loss ~ 0
    perfect = torch.zeros(B, P + L, V)
    for b in range(B):
        for i in range(L):
            perfect[b, P + i, response_ids[b, i]] = 30.0
    results.append(check(
        "loss: ~0 when the response is predicted perfectly",
        sft_loss(perfect, response_ids, resp_mask, t).item() < 1e-2))

    # loss must ignore the prompt span entirely: scrambling prompt logits
    # leaves the loss unchanged
    scrambled = logits.detach().clone()
    scrambled[:, :P, :] = torch.randn(B, P, V)
    results.append(check(
        "loss: invariant to prompt-span logits (response-only)",
        torch.allclose(sft_loss(logits.detach(), response_ids, resp_mask, t),
                       sft_loss(scrambled, response_ids, resp_mask, t))))

    # no masked response tokens -> no training signal -> loss is exactly 0
    empty_mask = torch.zeros(B, P + L, dtype=torch.bool)
    results.append(check(
        "loss: exactly 0 when nothing in the response is masked",
        sft_loss(logits.detach(), response_ids, empty_mask, t).item() == 0.0))

    # gradient flows back to the logits
    loss.backward()
    results.append(check(
        "loss: gradient flows to logits",
        logits.grad is not None and torch.isfinite(logits.grad).all().item()))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        sys.exit(0)
    print(f"{sum(results)}/{len(results)} passed — FAILURES ABOVE")
    sys.exit(1)


if __name__ == "__main__":
    main()
