#!/usr/bin/env python3
"""
Plot RULER accuracy vs KeyDiff reconstruction percentage for different KV sizes.
Generates two graphs: one for Llama and one for Qwen.
Run: python plot_reconstruction.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})
from matplotlib.ticker import FuncFormatter

# =============================================================================
# DATA FROM reconstruction_percentages.tex
# =============================================================================

# Format: { kv_size: { reconstruction_percentage: avg_score, ... }, ... }

# Qwen3-8B data
QWEN_DATA = {
    r"KV Size $2\%$": {
        1: 46.40,
        2: 49.05,
        5: 49.98,
        10: 52.46,
        25: 49.88,
    },
    r"KV Size $5\%$": {
        5: 65.73,
        10: 66.17,
        25: 68.09,
        50: 64.63,
        75: 54.61,
    },
    r"KV Size $10\%$": {
        10: 73.89,
        50: 83.79,
        75: 87.02,
        100: 86.40,
    },
}
QWEN_FULL = 94.66  # Full KV cache baseline

# Llama-3.1-8B-Instruct data
LLAMA_DATA = {
    r"KV Size $2\%$": {
        1: 44.07,
        2: 53.36,
        5: 60.96,
        10: 62.77,
        25: 59.88,
        50: 35.74,
    },
    r"KV Size $5\%$": {
        2: 64.10,
        5: 67.22,
        10: 69.39,
        25: 72.55,
        50: 72.87,
        75: 68.96,
        100: 62.84,
    },
    r"KV Size $10\%$": {
        50: 85.42,
        75: 88.11,
        90: 89.69,
        100: 89.71,
    },
}
LLAMA_FULL = 95.61  # Full KV cache baseline

# =============================================================================
# PLOT SETTINGS
# =============================================================================

FIGSIZE = (10, 6)
XLABEL = "Reconstruction Percentage"
YLABEL = "Score"

# Colors for each KV size
COLORS = {
    r"KV Size $2\%$": "#E74C3C",   # Red
    r"KV Size $5\%$": "#27AE60",   # Green
    r"KV Size $10\%$": "#4A90D9",  # Blue
}
MARKERS = {
    r"KV Size $2\%$": "o",
    r"KV Size $5\%$": "o",
    r"KV Size $10\%$": "o",
}

# =============================================================================
# PLOTTING CODE
# =============================================================================


def plot_model(data, full_score, title, output_file):
    """Plot reconstruction percentage vs RULER accuracy for a single model."""
    plt.figure(figsize=FIGSIZE)
    
    for kv_size, scores in data.items():
        # Sort by reconstruction percentage
        sorted_items = sorted(scores.items())
        x_vals = [item[0] for item in sorted_items]
        y_vals = [item[1] for item in sorted_items]
        
        color = COLORS.get(kv_size, "#000000")
        marker = MARKERS.get(kv_size, "o")
        
        plt.plot(
            x_vals, 
            y_vals, 
            label=kv_size,
            color=color,
            marker=marker,
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
    
    # Format x-axis as percentages
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}\\%"))
    
    # Set axis limits
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    
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
        output_file="reconstruction_qwen.pdf",
    )
    
    # Plot Llama graph
    plot_model(
        data=LLAMA_DATA,
        full_score=LLAMA_FULL,
        title="RULER 4k - Llama-3.1-8B-Instruct",
        output_file="reconstruction_llama.pdf",
    )


if __name__ == "__main__":
    main()
