# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})

# =============================================================================
# DATA
# =============================================================================

chunk_sizes = ["0.5k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

# Compute times ("OOM" = out of memory, 0.0 = method not applicable)
KVZIP_TIMES        = [89, 76, 68, 71, 96, "OOM", "OOM", "OOM", "OOM"]
KV2_05_TIMES       = [56,  47,  43,  44,  46, 53, "OOM", "OOM", "OOM"]
KV2_002_TIMES      = [42,  32,  26,  24,  23,   24,   24,   24,   25]
KV2_002_2IT_TIMES  = [62,  42,  32,  28,  26,   25,   26,   27,   32]
KV2_002_5IT_TIMES  = [130,  77,  50,  40,  36,   36,   40,   48,   72]

# Colors (muted for thesis style)
C_KVZIP   = "#E07B6A"   # Muted red
C_KV2_05  = "#6AAED6"   # Muted blue
C_KV2_002 = "#5BAD7A"   # Muted green
C_2IT     = "#1A7A3C"   # Deeper forest green
C_5IT     = "#002D14"   # Very dark green

FILL_ALPHA = 0.75       # Less transparent for cleaner look

BAR_WIDTH  = 0.25
BAR_GAP    = 0.0        # no gap between adjacent bars
FIGSIZE    = (9, 5)

# =============================================================================
# HELPERS
# =============================================================================

def process(data):
    """Return (numeric_values, oom_indices). OOM → 0, 0.0 → kept as-is."""
    nums, ooms = [], []
    for i, v in enumerate(data):
        if v == "OOM":
            nums.append(0)
            ooms.append(i)
        else:
            nums.append(float(v))
    return nums, ooms


def draw_bar(ax, x, offset, data, fill_color, label):
    centers = x + offset
    ax.bar(centers, data, BAR_WIDTH,
           color=(*mpl.colors.to_rgb(fill_color), FILL_ALPHA),
           edgecolor="white", linewidth=0.5, label=label)


def finish_ax(ax, ylim=150, sizes=None):
    if sizes is None:
        sizes = chunk_sizes
    x = np.arange(len(sizes))
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=22)
    ax.set_xlabel("Chunk size", fontsize=28)
    ax.set_ylabel("Compute time (s)", fontsize=28)
    ax.set_ylim(0, ylim)
    ax.axhline(y=19, color="black", linestyle=":", linewidth=1.5, label="_nolegend_")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#aaaaaa")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", labelsize=22)
    ax.legend(loc="upper center", fontsize=22, ncol=4,
              frameon=True, framealpha=0.9, edgecolor="#cccccc")
    plt.tight_layout()

# =============================================================================
# PLOT 1 — KVzip  vs  KV² 0.5  vs  KV² 0.02
# =============================================================================

def plot_comparison():
    x = np.arange(len(chunk_sizes))
    d1, _ = process(KVZIP_TIMES)
    d2, _ = process(KV2_05_TIMES)
    d3, _ = process(KV2_002_TIMES)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    step = BAR_WIDTH + BAR_GAP
    draw_bar(ax, x, -step, d1, C_KVZIP,   label="KVzip")
    draw_bar(ax, x,  0,    d2, C_KV2_05,  label=r"KV$^2$ (0.5)")
    draw_bar(ax, x, +step, d3, C_KV2_002, label=r"KV$^2$ (0.02)")

    finish_ax(ax, ylim=140)
    plt.savefig("chunk_size.pdf", bbox_inches="tight")
    print("Saved: chunk_size.pdf")
    plt.show()


# =============================================================================
# PLOT 2 — KV² 0.02  vs  KV² 0.02 + 2-iter  vs  KV² 0.02 + 5-iter
# =============================================================================

def plot_iterative():
    sizes = chunk_sizes
    x = np.arange(len(sizes))
    d3, _  = process(KV2_002_TIMES)
    d4, _  = process(KV2_002_2IT_TIMES)
    d5, _  = process(KV2_002_5IT_TIMES)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    step = BAR_WIDTH + BAR_GAP
    draw_bar(ax, x, -step, d3, C_KV2_002, label=r"KV$^2$")
    draw_bar(ax, x,  0,    d4, C_2IT,     label=r"KV$^2$ (2$\times$)")
    draw_bar(ax, x, +step, d5, C_5IT,     label=r"KV$^2$ (5$\times$)")

    finish_ax(ax, ylim=140, sizes=sizes)
    plt.savefig("chunk_size_iter.pdf", bbox_inches="tight")
    print("Saved: chunk_size_iter.pdf")
    plt.show()


# =============================================================================

if __name__ == "__main__":
    plot_comparison()
    plot_iterative()


# For KVzip specifically, this is what the measurement captures:
# After the initial prefill (where all 124k tokens are processed and the full KV cache is in memory at 32.37 GB), KVzip runs a second phase: it feeds chunks of the original context back through the model again ("repeat the previous context") to compute importance scores for each KV pair. These reconstruction forward passes are what the chunk_size parameter controls.
# During each reconstruction forward pass:
# The full KV cache is still in memory (32.37 GB settled — model weights + all stored KV pairs)
# On top of that, attention computations for the chunk are computed, needing temporary tensors proportional to chunk_size × context_length
# So:
# 32.37 GB = baseline = model weights + full KV cache sitting in memory while compression runs
# +1.54 GB overhead = peak extra memory during the reconstruction forward pass for that chunk size
# 33.91 GB total = the actual peak the GPU sees during the KVzip compression phase
# For the caption you could write something like:
# > Peak GPU memory during KVzip's context reconstruction phase as a function of chunk size. The dotted line (32.37 GB) marks the memory occupied by model weights and the full KV cache before compression. The overhead above the baseline reflects temporary attention tensors allocated during each reconstruction forward pass, which scale with chunk size.