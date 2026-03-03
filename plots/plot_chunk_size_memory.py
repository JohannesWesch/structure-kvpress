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

# Peak memory in GB ("OOM" = out of memory, 0.0 = method not applicable)
KVZIP_MEM        = [33.40, 33.37,  33.91,  37.57,  51.42, "OOM", "OOM", "OOM", "OOM"]
KV2_05_MEM       = [33.36,  33.39,  33.46,  34.43,  39.68,  59.94, "OOM", "OOM", "OOM"]
KV2_002_MEM      = [33.33,  33.33,  33.34,  33.34,  33.35,   33.37,   35.42,   44.02,   73.40]
KV2_002_2IT_MEM  = [33.40,  33.40,  33.40,  33.41,  33.42,   33.44,   35.49,   44.09,   73.49]
KV2_002_5IT_MEM  = [33.59,  33.59,  33.59,  33.60,  33.61,   33.63,   35.68,   44.30,   73.75]

SETTLED_MEM = 32.37  # GB — model + KV cache after prefill (baseline before compression runs)

# Colors (muted for thesis style)
C_KVZIP   = "#E07B6A"   # Muted red
C_KV2_05  = "#6AAED6"   # Muted blue
C_KV2_002 = "#5BAD7A"   # Muted green
C_2IT     = "#1A7A3C"   # Deeper forest green
C_5IT     = "#002D14"   # Very dark green

FILL_ALPHA = 0.75
BAR_WIDTH  = 0.25
BAR_GAP    = 0.0
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


def finish_ax(ax, ylim=80, sizes=None):
    if sizes is None:
        sizes = chunk_sizes
    x = np.arange(len(sizes))
    ax.set_xticks(x)
    ax.set_xticklabels(sizes, fontsize=10)
    ax.set_xlabel("Chunk size", fontsize=12)
    ax.set_ylabel("Peak memory (GB)", fontsize=12)
    ax.set_ylim(0, ylim)
    ax.axhline(y=SETTLED_MEM, color="black", linestyle=":", linewidth=1.5, label="_nolegend_")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#aaaaaa")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(loc="upper center", fontsize=10, ncol=4,
              frameon=True, framealpha=0.9, edgecolor="#cccccc")
    plt.tight_layout()

# =============================================================================
# PLOT 1 — KVzip  vs  KV² 0.5  vs  KV² 0.02
# =============================================================================

def plot_comparison():
    x = np.arange(len(chunk_sizes))
    d1, _ = process(KVZIP_MEM)
    d2, _ = process(KV2_05_MEM)
    d3, _ = process(KV2_002_MEM)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    step = BAR_WIDTH + BAR_GAP
    draw_bar(ax, x, -step, d1, C_KVZIP,   label="KVzip")
    draw_bar(ax, x,  0,    d2, C_KV2_05,  label=r"KV$^2$ (0.5)")
    draw_bar(ax, x, +step, d3, C_KV2_002, label=r"KV$^2$ (0.02)")

    finish_ax(ax, ylim=80)
    plt.savefig("chunk_size_memory.pdf", bbox_inches="tight")
    print("Saved: chunk_size_memory.pdf")
    plt.show()


# =============================================================================
# PLOT 2 — KV² 0.02  vs  KV² 0.02 + 2-iter  vs  KV² 0.02 + 5-iter
# =============================================================================

def plot_iterative():
    sizes = chunk_sizes
    x = np.arange(len(sizes))
    d3, _ = process(KV2_002_MEM)
    d4, _ = process(KV2_002_2IT_MEM)
    d5, _ = process(KV2_002_5IT_MEM)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    step = BAR_WIDTH + BAR_GAP
    draw_bar(ax, x, -step, d3, C_KV2_002, label=r"KV$^2$ (1$\times$)")
    draw_bar(ax, x,  0,    d4, C_2IT,     label=r"KV$^2$ (2$\times$)")
    draw_bar(ax, x, +step, d5, C_5IT,     label=r"KV$^2$ (5$\times$)")

    finish_ax(ax, ylim=80, sizes=sizes)
    plt.savefig("chunk_size_memory_iter.pdf", bbox_inches="tight")
    print("Saved: chunk_size_memory_iter.pdf")
    plt.show()


# =============================================================================

if __name__ == "__main__":
    plot_comparison()
    plot_iterative()
