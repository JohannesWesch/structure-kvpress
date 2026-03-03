#!/usr/bin/env python3
"""
Plot RULER accuracy vs KV cache size for different methods.
Generates two graphs: one for Llama and one for Qwen.
Run: python plot_ruler_methods.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})

# =============================================================================
# DATA FROM ruler.tex
# =============================================================================

# Format: { "Method Name": { kv_size: avg_score, ... }, ... }

# Qwen3-8B data
QWEN_DATA = {
    r"KV$^2$": {
        0.02: 49.05,
        0.05: 65.73,
        0.10: 73.89,
    },
    "KVzip": {
        0.02: 18.08,
        0.05: 36.04,
        0.10: 86.94,
    },
    "KeyDiff": {
        0.02: 10.37,
        0.05: 19.73,
        0.10: 37.15,
    },
    "ExpectedAttention": {
        0.02: 7.48,
        0.05: 36.80,
        0.10: 63.60,
    },
}
QWEN_FULL = 94.66  # Full KV cache baseline

# Llama-3.1-8B-Instruct data
LLAMA_DATA = {
    r"KV$^2$": {
        0.02: 53.36,
        0.05: 67.22,
        0.10: 76.35,
    },
    "KVzip": {
        0.02: 17.93,
        0.05: 63.37,
        0.10: 91.14,
    },
    "KeyDiff": {
        0.02: 23.22,
        0.05: 47.11,
        0.10: 60.43,
    },
    "ExpectedAttention": {
        0.02: 7.22,
        0.05: 12.74,
        0.10: 30.10,
    },
}
LLAMA_FULL = 95.61  # Full KV cache baseline

# =============================================================================
# PLOT SETTINGS
# =============================================================================

FIGSIZE = (10, 6)
XLABEL = "Compression Ratio"
YLABEL = "Score"

# Colors for each method
COLORS = {
    "KeyDiff": "#000000",           # Black
    "ExpectedAttention": "#9B59B6", # Purple
    "KVzip": "#E74C3C",             # Red
    r"KV$^2$": "#4A90D9",            # Blue
}

# =============================================================================
# PLOTTING CODE
# =============================================================================


def plot_model(data, full_score, title, output_file):
    """Plot KV size vs RULER accuracy for a single model."""
    plt.figure(figsize=FIGSIZE)
    
    for method, scores in data.items():
        # Sort by KV size (descending for x-axis from high to low)
        sorted_items = sorted(scores.items(), reverse=True)
        x_vals = [item[0] for item in sorted_items]
        y_vals = [item[1] for item in sorted_items]
        
        color = COLORS.get(method, "#000000")
        
        plt.plot(
            x_vals, 
            y_vals, 
            label=method,
            color=color,
            marker="o",
            markersize=8,
            linewidth=2,
        )
    
    # Add baseline (no compression) horizontal dotted line
    plt.axhline(y=full_score, color='gray', linestyle='--', linewidth=1.5, label='Full KV cache')
    
    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    ax = plt.gca()

    # Invert x-axis so it goes from 0.10 to 0.02
    plt.xlim(0.11, 0.01)
    plt.ylim(0, 100)

    ax.set_xticks([0.10, 0.05, 0.02])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Saved plot to: {output_file}")
    
    plt.show()


def main():
    # Plot Qwen graph
    plot_model(
        data=QWEN_DATA,
        full_score=QWEN_FULL,
        title="RULER 4k - Qwen3-8B",
        output_file="ruler_4k_qwen.pdf",
    )
    
    # Plot Llama graph
    plot_model(
        data=LLAMA_DATA,
        full_score=LLAMA_FULL,
        title="RULER 4k - Llama-3.1-8B-Instruct",
        output_file="ruler_4k_llama.pdf",
    )


if __name__ == "__main__":
    main()
