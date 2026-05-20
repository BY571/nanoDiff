"""Tests for the SFT forward process, loss, and data encoding (nanodiff/sft.py).

Same standalone-assertion style as smoke_test.py — run directly, exits
non-zero on any failure so it doubles as a CI gate.

    uv run python test_sft.py
"""
import sys

import tiktoken
import torch

from nanodiff.sft import sft_forward_process, sft_loss, encode_sft_example

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

    # ---- encode_sft_example -----------------------------------------------
    enc = tiktoken.get_encoding("gpt2")
    EOT = 50256
    PL, RL = 64, 96

    # short example, no input
    p, r = encode_sft_example("Name a color.", "", "Blue.", enc, PL, RL, EOT)
    results.append(check(
        "encode: prompt/response have the fixed lengths",
        len(p) == PL and len(r) == RL))

    results.append(check(
        "encode: short prompt is LEFT-padded with EOT",
        p[0] == EOT and p[-1] != EOT))

    results.append(check(
        "encode: short response is RIGHT-padded with EOT",
        r[-1] == EOT and r[0] != EOT))

    results.append(check(
        "encode: prompt carries the instruction text",
        "Name a color." in enc.decode(p)))

    results.append(check(
        "encode: prompt ends with the '### Response:' generation cue",
        enc.decode(p).rstrip().endswith("### Response:")))

    results.append(check(
        "encode: response begins with the output text",
        enc.decode(r).startswith("Blue.")))

    # the answer is followed by an EOT end-marker before the padding
    results.append(check(
        "encode: an EOT end-marker follows the answer",
        EOT in r and r.index(EOT) == len(enc.encode("Blue."))))

    # example WITH an input field
    p2, _ = encode_sft_example("Translate to French.", "Hello world",
                               "Bonjour le monde.", enc, PL, RL, EOT)
    results.append(check(
        "encode: the input field appears in the prompt",
        "Hello world" in enc.decode(p2)))

    # long output -> response truncated from the right, exact length kept
    long_out = " ".join(["word"] * 500)
    _, r3 = encode_sft_example("Say words.", "", long_out, enc, PL, RL, EOT)
    results.append(check(
        "encode: over-long response is truncated to exactly response_len",
        len(r3) == RL and enc.decode(r3).startswith("word word")))

    # long prompt -> truncated from the left, the '### Response:' cue survives
    long_instr = " ".join(["explain"] * 500)
    p4, _ = encode_sft_example(long_instr, "", "ok", enc, PL, RL, EOT)
    results.append(check(
        "encode: over-long prompt truncates from the left, keeps the cue",
        len(p4) == PL and enc.decode(p4).rstrip().endswith("### Response:")))

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    print(f"{sum(not r for r in results)}/{len(results)} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
