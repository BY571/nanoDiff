"""LAMBADA scaling figure for nanoDiff.

Left panel:  LAMBADA accuracy vs model parameters, base vs SFT.
Right panel: LAMBADA last-word PPL vs model parameters (log y), base vs SFT.

The "alignment tax" (base→SFT drop) becomes visible as the vertical
distance between the base and SFT lines at each parameter scale, and
grows visibly at 350M, matching the documented finding.

Pure static-numbers plot, no wandb needed. From the repo root:
    python scripts/plot_lambada.py
Writes to assets/lambada.png if run from the repo root, otherwise /tmp/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Hardcoded measured numbers (also in README and HF model cards)
DATA = {
    #          (params,  base_acc, base_ppl, sft_acc, sft_ppl)
    "50M":    (50e6,   19.83,   834.0,   14.32, 3344.0),
    "150M":   (150e6,  21.89,   358.0,   15.74, 1606.0),
    "350M":   (350e6,  36.95,    55.3,   25.13,  286.6),
}

C_BASE = "#c44e52"   # red (matches the 350M base color in the scaling plot)
C_SFT  = "#4c72b0"   # blue

names = list(DATA.keys())
params = [DATA[n][0] for n in names]
base_acc = [DATA[n][1] for n in names]
base_ppl = [DATA[n][2] for n in names]
sft_acc  = [DATA[n][3] for n in names]
sft_ppl  = [DATA[n][4] for n in names]

fig, (ax_acc, ax_ppl) = plt.subplots(1, 2, figsize=(13, 5))

# Left: accuracy
ax_acc.plot(params, base_acc, "-o", color=C_BASE, label="base", markersize=12,
            markeredgecolor="black", markeredgewidth=1, linewidth=2)
ax_acc.plot(params, sft_acc,  "-o", color=C_SFT,  label="SFT (Alpaca-cleaned)",
            markersize=12, markeredgecolor="black", markeredgewidth=1, linewidth=2)

# Annotate every point with its number; offset SFT down, base up
for i, n in enumerate(names):
    ax_acc.annotate(f"{base_acc[i]:.1f}%", (params[i], base_acc[i]),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=10, color=C_BASE, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85))
    ax_acc.annotate(f"{sft_acc[i]:.1f}%", (params[i], sft_acc[i]),
                    textcoords="offset points", xytext=(10, -18),
                    fontsize=10, color=C_SFT, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85))

# Show alignment-tax pp drop as a small vertical bracket at 350M
ax_acc.annotate("", xy=(params[2] * 1.18, sft_acc[2]),
                xytext=(params[2] * 1.18, base_acc[2]),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.2))
ax_acc.text(params[2] * 1.32, (base_acc[2] + sft_acc[2]) / 2,
            "−11.8 pp\n(32% tax)",
            fontsize=9, color="gray", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="lightgray", alpha=0.9))

ax_acc.set_xscale("log")
ax_acc.set_xlabel("Model parameters (M, log scale)")
ax_acc.set_ylabel("LAMBADA accuracy (%)")
ax_acc.set_title("LAMBADA accuracy vs model size")
ax_acc.set_ylim(10, 42)
ax_acc.set_xlim(35e6, 700e6)
ax_acc.grid(True, alpha=0.3)
ax_acc.legend(loc="upper left", framealpha=0.95)

# label the x positions clearly: rather than raw param counts on log axis,
# annotate "50M", "150M", "350M" below the axis
ax_acc.set_xticks(params)
ax_acc.set_xticklabels(["50M", "150M", "350M"])

# Right: PPL on log y-axis
ax_ppl.plot(params, base_ppl, "-o", color=C_BASE, label="base", markersize=12,
            markeredgecolor="black", markeredgewidth=1, linewidth=2)
ax_ppl.plot(params, sft_ppl,  "-o", color=C_SFT,  label="SFT (Alpaca-cleaned)",
            markersize=12, markeredgecolor="black", markeredgewidth=1, linewidth=2)

for i, n in enumerate(names):
    ax_ppl.annotate(f"{base_ppl[i]:.0f}", (params[i], base_ppl[i]),
                    textcoords="offset points", xytext=(10, 6),
                    fontsize=10, color=C_BASE, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85))
    ax_ppl.annotate(f"{sft_ppl[i]:.0f}", (params[i], sft_ppl[i]),
                    textcoords="offset points", xytext=(10, -18),
                    fontsize=10, color=C_SFT, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="none", alpha=0.85))

ax_ppl.set_xscale("log")
ax_ppl.set_yscale("log")
ax_ppl.set_xlabel("Model parameters (M, log scale)")
ax_ppl.set_ylabel("LAMBADA perplexity (log scale)")
ax_ppl.set_title("LAMBADA last-word PPL vs model size")
ax_ppl.set_xticks(params)
ax_ppl.set_xticklabels(["50M", "150M", "350M"])
ax_ppl.set_xlim(35e6, 700e6)
ax_ppl.grid(True, alpha=0.3, which="both")
ax_ppl.legend(loc="upper right", framealpha=0.95)

fig.suptitle("nanoDiff LAMBADA: base capability vs SFT alignment tax",
             fontsize=13, weight="bold")
plt.tight_layout()

out_path = "assets/lambada.png" if os.path.isdir("assets") else "/tmp/lambada.png"
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
plt.savefig(out_path, dpi=160, bbox_inches="tight")
print(f"saved {out_path}")
