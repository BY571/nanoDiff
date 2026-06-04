"""Build a 2-panel scaling figure from wandb training history.

Left panel:  validation NLL vs training tokens seen (training curves
             for the 4 base runs: 50M @ 2B, 50M @ 3B control, 150M @ 3B,
             350M @ 10B).
Right panel: final validation NLL vs model parameters (log x-axis), the
             scaling law as endpoints. Colors carry over from the left
             panel so the eye links the trajectory to its endpoint.

Requires `wandb` (with API auth via ~/.netrc or WANDB_API_KEY) and
`matplotlib`. From the repo root:
    python scripts/plot_scaling.py
Writes to assets/scaling.png if run from the repo root, otherwise /tmp/.
"""
import os
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Map run-name → (label, params, tokens_per_iter)
# tokens_per_iter = batch_size × grad_accum_steps × block_size, pulled from
# the matching pretrain/configs/*.py at training time.
FAMILY = {
    "nanodiff-50m-spark":  ("50M @ 2B (original)",          50e6,  128 * 1 * 1024),
    "nanodiff-50m-3b":     ("50M @ 3B (matched control)",   50e6,  128 * 1 * 512),
    "nanodiff-150m":       ("150M @ 3B",                    150e6, 128 * 1 * 512),
    "nanodiff-350m":       ("350M @ 10B",                   350e6, 64 * 4 * 512),
}

# Consistent color per model size, base palette
COLORS = {
    "nanodiff-50m-spark": "#4c72b0",  # blue
    "nanodiff-50m-3b":    "#7aa6cf",  # lighter blue (the control)
    "nanodiff-150m":      "#dd8452",  # orange
    "nanodiff-350m":      "#c44e52",  # red
}

api = wandb.Api()
print("listing runs in sebastian-dittert/nanodiff …")
runs = list(api.runs("sebastian-dittert/nanodiff"))
print(f"  found {len(runs)} total runs in project")

scaling_endpoints = []  # (params, final_val, label, color)

fig, (ax_curves, ax_law) = plt.subplots(1, 2, figsize=(13, 5))

for run in runs:
    if run.name not in FAMILY:
        continue
    label, params, tokens_per_iter = FAMILY[run.name]
    color = COLORS[run.name]

    # Download the val/loss history. `samples` cap is generous; the runs
    # log val every eval_interval iters (1000-2000), so this is plenty.
    history = run.history(keys=["val/loss", "_step"], samples=10000, pandas=True)
    history = history.dropna(subset=["val/loss"])
    if history.empty:
        print(f"  SKIP {run.name}: no val/loss history")
        continue

    tokens_seen_b = history["_step"] * tokens_per_iter / 1e9
    val_loss = history["val/loss"]

    ax_curves.plot(tokens_seen_b, val_loss, "-o", label=label,
                   linewidth=2, color=color, markersize=4,
                   markeredgecolor=color, markerfacecolor=color)

    final_val = float(val_loss.iloc[-1])
    final_tokens = float(tokens_seen_b.iloc[-1])
    scaling_endpoints.append((params, final_val, label, color))
    print(f"  {run.name:18s}: {len(history)} val points, "
          f"final {final_val:.3f} nats at {final_tokens:.1f}B tokens")

# --- Left panel: training curves ---
ax_curves.set_xlabel("Training tokens seen (B, log scale)")
ax_curves.set_ylabel("Validation NLL (nats/token)")
ax_curves.set_title("Training dynamics: val NLL vs tokens")
ax_curves.set_xscale("log")
ax_curves.set_ylim(3.3, 4.5)   # focus tightly on the converged regime where the scaling story is visible
ax_curves.set_xlim(0.2, 12)    # clip the first hundred million tokens where everything is still very high
ax_curves.grid(True, alpha=0.3)
ax_curves.legend(loc="upper right", framealpha=0.95)

# --- Right panel: scaling law (final val NLL vs params) ---
for params, final_val, label, color in sorted(scaling_endpoints, key=lambda x: x[0]):
    ax_law.scatter(params / 1e6, final_val, s=140, color=color,
                   edgecolors="black", linewidths=1, zorder=3)
    # Distinguish the two 50M points (original vs matched control)
    if "(matched control)" in label:
        text = "50M (3B)"
        offset = (10, -14)   # below the dot to separate from the 2B point
    elif "(original)" in label:
        text = "50M (2B)"
        offset = (10, 8)     # above the dot
    else:
        text = label.split(" @ ")[0]
        offset = (10, 6)
    ax_law.annotate(text, (params / 1e6, final_val),
                    textcoords="offset points", xytext=offset,
                    fontsize=10, color=color, weight="bold")

# Draw a trend line through the three primary scales (50M @ 2B, 150M, 350M)
# in log-NLL vs log-params space, for visual anchoring
primary = [(p, v) for p, v, l, c in scaling_endpoints if "matched control" not in l]
if len(primary) >= 2:
    ps = np.array([p for p, v in primary])
    vs = np.array([v for p, v in primary])
    log_ps = np.log10(ps)
    fit = np.polyfit(log_ps, vs, 1)
    fit_x = np.logspace(np.log10(ps.min() * 0.7), np.log10(ps.max() * 1.4), 100)
    fit_y = fit[0] * np.log10(fit_x) + fit[1]
    ax_law.plot(fit_x / 1e6, fit_y, "--", color="gray", alpha=0.6,
                label=f"trend (slope {fit[0]:.2f} nats/decade)")
    ax_law.legend(loc="upper right", framealpha=0.95)

ax_law.set_xlabel("Model parameters (M, log scale)")
ax_law.set_ylabel("Final validation NLL (nats/token)")
ax_law.set_title("Scaling law: final val NLL vs model size")
ax_law.set_xscale("log")
ax_law.grid(True, alpha=0.3)

fig.suptitle("nanoDiff scaling: 50M → 150M → 350M base models on FineWeb-Edu",
             fontsize=13, weight="bold")
plt.tight_layout()

out_path = "assets/scaling.png" if os.path.isdir("assets") else "/tmp/scaling.png"
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"\nsaved {out_path}")
