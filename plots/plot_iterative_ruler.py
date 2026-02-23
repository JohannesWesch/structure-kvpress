#!/usr/bin/env python3
"""
Plot RULER accuracy vs number of self-refinement iterations (1–5).
Data from the second table in iterative_ruler.tex (Random 2% and Random 10%).
Four lines: KV²+ and KV² for 2% and 10%; blue for 2%, green for 10%;
KV²+ solid, KV² dotted.
Run: python plot_iterative_ruler.py
"""

import matplotlib.pyplot as plt

# =============================================================================
# DATA FROM iterative_ruler.tex (second table: Self-Refinement Steps)
# =============================================================================
# X = iterations 1–5, Y = Avg. (RULER accuracy %)

ITERATIONS = [1, 2, 3, 4, 5]

# Random 2%: KV² and KV²+
KV2_2PCT = [17.3, 32.51, 36.91, 39.90, 40.30]
KV2_PLUS_2PCT = [18.09, 40.09, 43.68, 47.72, 49.61]

# Random 10%: KV² and KV²+
KV2_10PCT = [20.17, 35.28, 38.25, 39.25, 42.06]
KV2_PLUS_10PCT = [21.92, 41.06, 46.64, 48.79, 50.73]

FULL_SCORE = 95.61  # Full KV cache baseline (Llama-3.1-8B-Instruct)

# =============================================================================
# PLOT SETTINGS
# =============================================================================

FIGSIZE = (10, 6)
XLABEL = "Iterations"
YLABEL = "RULER Accuracy (%)"

# Blue for 2%, green for 10%
COLOR_2PCT = "#4A90D9"
COLOR_10PCT = "#27AE60"

# =============================================================================
# PLOTTING
# =============================================================================


def main():
    plt.figure(figsize=FIGSIZE)

    # KV²+ 2% — solid blue
    plt.plot(
        ITERATIONS,
        KV2_PLUS_2PCT,
        label="KV²+ (2%)",
        color=COLOR_2PCT,
        linestyle="-",
        marker="o",
        markersize=8,
        linewidth=2,
    )
    # KV² 2% — dotted blue
    plt.plot(
        ITERATIONS,
        KV2_2PCT,
        label="KV² (2%)",
        color=COLOR_2PCT,
        linestyle=":",
        marker="o",
        markersize=8,
        linewidth=2,
    )
    # KV²+ 10% — solid green
    plt.plot(
        ITERATIONS,
        KV2_PLUS_10PCT,
        label="KV²+ (10%)",
        color=COLOR_10PCT,
        linestyle="-",
        marker="o",
        markersize=8,
        linewidth=2,
    )
    # KV² 10% — dotted green
    plt.plot(
        ITERATIONS,
        KV2_10PCT,
        label="KV² (10%)",
        color=COLOR_10PCT,
        linestyle=":",
        marker="o",
        markersize=8,
        linewidth=2,
    )

    plt.axhline(
        y=FULL_SCORE,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Full KV cache",
    )

    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title(
        "RULER-4k: Self-Refinement Steps (Llama-3.1-8B-Instruct)\nBase press: Random",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 5.5)
    plt.ylim(0, 60)
    plt.xticks(ITERATIONS)

    plt.tight_layout()
    plt.savefig("iterative_ruler.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: iterative_ruler.png")
    plt.show()


if __name__ == "__main__":
    main()
