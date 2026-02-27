#!/usr/bin/env python3
"""
Plot comparison of KV cache compression methods.
Edit the data below and run: python plot_comparison.py
"""

import matplotlib.pyplot as plt

# =============================================================================
# EDIT YOUR DATA HERE
# =============================================================================

# Format: { "Method Name": { compression_ratio: score, ... }, ... }
DATA = {
    "KVzip": {
        0.0: 92.9,
        0.25: 92.76,
        0.5: 92.94,
        0.75: 93.02,
        0.9: 78.01,
        0.95: 38.16,
        0.98: 18.41,
        0.99: 17.15
    },
    "KV²": { # with kvsquared_2
        0.0: 92.9,
        0.25: 92.9,
        0.5: 92.9,
        0.75: 92.87,
        0.9: 77.07,
        0.95: 69.66,
        0.98: 64.95,
        0.99: 57.16,
    },
    # "KV² 3+": { # with kvsquared_3+
    #     0.0: 92.9,
    #     0.25: 92.9,
    #     0.5: 92.9,
    #     0.98: 66.67,
    #     0.99: 61.48,
    # },
    "KeyDiff": {
        0.0: 92.9,
        0.25: 82.9,
        0.5: 74.5,
        0.75: 66.9,
        0.9: 53.1,
        0.95: 31.99,
        0.98: 14.13,
        0.99: 8.81,
    },
    "Expected Attention": {
        0.0: 92.9,
        0.25: 93.2,
        0.5: 92.7,
        0.75: 85.6,
        0.9: 62.7,
        0.95: 43.03,
        0.98: 15.42,
        0.99: 9.53,
    },

}

# =============================================================================
# PLOT SETTINGS
# =============================================================================

TITLE = "RULER 16k - Qwen3-8B"
XLABEL = "Compression Ratio"
YLABEL = "Score (%)"
FIGSIZE = (10, 6)
OUTPUT_FILE = "RULER_16k.png"  # Set to None to only display, not save

COLORS = {
    "KVzip": "#E74C3C",             # Red
    "KV²": "#4A90D9",      # Blue
    "KV² 3+": "#1B3A6B",      # Dark Blue
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
    tick_labels = [f"{int(r * 100)}%" for r in all_ratios]

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

    plt.axhline(y=92.9, color='gray', linestyle='--', linewidth=1.5, label='No compression')

    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title(TITLE, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.set_xticks(range(len(all_ratios)))
    ax.set_xticklabels(tick_labels)

    plt.xlim(0, len(all_ratios) - 0.5)
    plt.ylim(0, 100)

    plt.tight_layout()

    if OUTPUT_FILE:
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {OUTPUT_FILE}")

    plt.show()


if __name__ == "__main__":
    plot_comparison()
