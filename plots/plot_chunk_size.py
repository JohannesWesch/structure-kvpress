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
KVZIP_TIMES        = [138, 131, 116, 121, 132, "OOM", "OOM", "OOM", "OOM"]
KV2_05_TIMES       = [108,  80,  72,  70,  73,    80, "OOM", "OOM", "OOM"]
KV2_002_TIMES      = [ 98,  60,  41,  32,  27,    25,    25,    25,    26]
KV2_002_2IT_TIMES  = [0.0, 0.0, 0.0,  43,  34,    30,    30,    31,    35]
KV2_002_5IT_TIMES  = [0.0, 0.0, 0.0,  77,  54,    47,    49,    55,    78]

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
    ax.set_xticklabels(sizes, fontsize=10)
    ax.set_xlabel("Repeat chunk size", fontsize=12)
    ax.set_ylabel("Compute time (s)", fontsize=12)
    ax.set_ylim(0, ylim)
    ax.axhline(y=20, color="black", linestyle=":", linewidth=1.5, label="Prefill")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#aaaaaa")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    handles, labels = ax.get_legend_handles_labels()
    if "Prefill" in labels:
        i = labels.index("Prefill")
        handles.append(handles.pop(i))
        labels.append(labels.pop(i))
    ax.legend(handles, labels, loc="upper center", fontsize=10, ncol=4,
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

    finish_ax(ax, ylim=150)
    plt.savefig("chunk_size.pdf", bbox_inches="tight")
    print("Saved: chunk_size.pdf")
    plt.show()


# =============================================================================
# PLOT 2 — KV² 0.02  vs  KV² 0.02 + 2-iter  vs  KV² 0.02 + 5-iter
# =============================================================================

def plot_iterative():
    start = chunk_sizes.index("4k")
    sizes = chunk_sizes[start:]
    x = np.arange(len(sizes))
    d3, _  = process(KV2_002_TIMES[start:])
    d4, _  = process(KV2_002_2IT_TIMES[start:])
    d5, _  = process(KV2_002_5IT_TIMES[start:])

    fig, ax = plt.subplots(figsize=FIGSIZE)

    step = BAR_WIDTH + BAR_GAP
    draw_bar(ax, x, -step, d3, C_KV2_002, label=r"KV$^2$ (1$\times$)")
    draw_bar(ax, x,  0,    d4, C_2IT,     label=r"KV$^2$ (2$\times$)")
    draw_bar(ax, x, +step, d5, C_5IT,     label=r"KV$^2$ (5$\times$)")

    finish_ax(ax, ylim=110, sizes=sizes)
    plt.savefig("chunk_size_iter.pdf", bbox_inches="tight")
    print("Saved: chunk_size_iter.pdf")
    plt.show()


# =============================================================================

if __name__ == "__main__":
    plot_comparison()
    plot_iterative()
