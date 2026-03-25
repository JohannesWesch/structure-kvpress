#!/usr/bin/env python3
"""
Plot comparison of KV cache compression methods.
Edit the data below and run: python plot_comparison.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})

# =============================================================================
# EDIT YOUR DATA HERE
# =============================================================================

# Format: { "Method Name": { compression_ratio: score, ... }, ... }
DATA = {
    r"KV$^2$": { # with kvsquared_2
        0.0: 93.01,
        0.25: 92.96,
        0.5: 93.09,
        0.75: 90.27,
        0.9: 76.18,
        0.95: 70.38,
        0.98: 62.19,
        0.99: 49.66,
        0.995: 26.9,
    },
    "KVzip": {
        0.0: 93.01,
        0.25: 92.95,
        0.5: 93.08,
        0.75: 92.64,
        0.9: 78.33,
        0.95: 39.13,
        0.98: 18.94,
        0.99: 18.92,
        0.995: 18.68,
    },
    
    # "KV² 3+": { # with kvsquared_3+
    #     0.0: 92.9,
    #     0.25: 92.9,
    #     0.5: 92.9,
    #     0.98: 66.67,
    #     0.99: 61.48,
    # },
    "KeyDiff": {
        0.0: 93.01,
        0.25: 83.1,
        0.5: 74.44,
        0.75: 66.88,
        0.9: 53.02,
        0.95: 33.14,
        0.98: 14.89,
        0.99: 8.81,
        0.995: 5.6,
    },
    "Expected Attention": {
        0.0: 93.01,
        0.25: 93.24,
        0.5: 92.63,
        0.75: 85.42,
        0.9: 62.82,
        0.95: 45.77,
        0.98: 16.5,
        0.99: 9.53,
        0.995: 4.11,
    },

}

# =============================================================================
# PLOT SETTINGS
# =============================================================================

TITLE = "RULER 16k - Qwen3-8B"
XLABEL = "Compression Ratio"
YLABEL = "Score"
FIGSIZE = (10, 6)
OUTPUT_FILE = "ruler_16k.pdf"  # Set to None to only display, not save

COLORS = {
    "KVzip": "#E74C3C",             # Red
    r"KV$^2$": "#4A90D9",      # Blue
    r"KV$^2$ 3+": "#1B3A6B",  # Dark Blue
    "KeyDiff": "#000000",           # Black
    "Expected Attention": "#9B59B6", # Purple
}

# =============================================================================
# PLOTTING CODE (no need to edit below)
# =============================================================================


def plot_comparison():
    # Collect all unique compression ratios across methods and assign equidistant positions
    all_ratios = sorted({r for scores in DATA.values() for r in scores})
    ratio_to_pos = {r: i for i, r in enumerate(all_ratios)}
    tick_labels = [str(r) for r in all_ratios]

    plt.figure(figsize=FIGSIZE)

    for method_name, scores in DATA.items():
        sorted_items = sorted(scores.items())
        x_vals = [ratio_to_pos[item[0]] for item in sorted_items]
        y_vals = [item[1] for item in sorted_items]

        color = COLORS.get(method_name, "#000000")

        plt.plot(
            x_vals,
            y_vals,
            label=method_name,
            color=color,
            marker="o",
            markersize=8,
            linewidth=2,
        )

    plt.axhline(y=93.01, color='gray', linestyle='--', linewidth=1.5, label='No compression')

    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title(TITLE, fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.set_xticks(range(len(all_ratios)))
    ax.set_xticklabels(tick_labels)

    plt.xlim(0, len(all_ratios) - 0.5)
    plt.ylim(0, 100)

    plt.tight_layout()

    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, bbox_inches="tight")
        print(f"Saved plot to: {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    plot_comparison()


# r"KV$^2$_normal": { # with kvsquared
#         0.0: 92.9,
#         0.25: 92.97,
#         0.5: 92.71,
#         0.75: 89.23,
#         0.9: 75.22,
#         0.95: 69.42,
#         0.98: 61.70,
#         0.99: 49.67,
#     },
#     r"KV$^2$_3+": { # with kvsquared_3+
#         0.0: 92.9,
#         0.25: 93.13,
#         0.5: 93.02,
#         0.75: 89.75,
#         0.9: 77.86,
#         0.95: 71.52,
#         0.98: 63.61,
#         0.99: 49.40,
#     },
