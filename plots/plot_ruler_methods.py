#!/usr/bin/env python3
"""
Plot RULER accuracy vs KV cache size for different methods.
Generates two graphs: one for Llama and one for Qwen.
Run: python plot_ruler_methods.py
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =============================================================================
# DATA FROM ruler.tex
# =============================================================================

# Format: { "Method Name": { kv_size: avg_score, ... }, ... }

# Qwen3-8B data
QWEN_DATA = {
    "KeyDiff": {
        2: 10.37,
        5: 19.73,
        10: 37.15,
    },
    "ExpectedAttention": {
        2: 7.48,
        5: 36.80,
        10: 63.60,
    },
    "KVzip": {
        2: 18.08,
        5: 36.04,
        10: 86.94,
    },
    "KV²": {
        2: 49.05,
        5: 65.73,
        10: 73.89,
    },
}
QWEN_FULL = 94.66  # Full KV cache baseline

# Llama-3.1-8B-Instruct data
LLAMA_DATA = {
    "KeyDiff": {
        2: 23.22,
        5: 47.11,
        10: 60.43,
    },
    "ExpectedAttention": {
        2: 7.22,
        5: 12.74,
        10: 30.10,
    },
    "KVzip": {
        2: 17.93,
        5: 63.37,
        10: 91.14,
    },
    "KV²": {
        2: 53.36,
        5: 67.22,
        10: 76.35,
    },
}
LLAMA_FULL = 95.61  # Full KV cache baseline

# =============================================================================
# PLOT SETTINGS
# =============================================================================

FIGSIZE = (10, 6)
XLABEL = "KV Cache Size"
YLABEL = "RULER Accuracy (%)"

# Colors for each method
COLORS = {
    "KeyDiff": "#000000",           # Black
    "ExpectedAttention": "#9B59B6", # Purple
    "KVzip": "#E74C3C",             # Red
    "KV²": "#4A90D9",               # Blue
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
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis as percentages
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
    
    # Invert x-axis so it goes from 15% to 0%
    plt.xlim(15, 0)
    plt.ylim(0, 100)
    
    # Set x-axis ticks at 10, 5, 2
    ax.set_xticks([10, 5, 2])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_file}")
    
    plt.show()


def main():
    # Plot Qwen graph
    plot_model(
        data=QWEN_DATA,
        full_score=QWEN_FULL,
        title="RULER-4k: Qwen3-8B",
        output_file="ruler_qwen.png",
    )
    
    # Plot Llama graph
    plot_model(
        data=LLAMA_DATA,
        full_score=LLAMA_FULL,
        title="RULER-4k: Llama-3.1-8B-Instruct",
        output_file="ruler_llama.png",
    )


if __name__ == "__main__":
    main()
