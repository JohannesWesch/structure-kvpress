#!/usr/bin/env python3
"""
Plot RULER accuracy vs number of self-refinement iterations (1–5).
Data from iterative_ruler.tex. Eight lines total:
  Random:  blue (2%), green (10%); KV²+ solid, KV² dotted
  KeyDiff: red  (2%), orange(10%); KV²+ solid, KV² dotted
Run: python plot_iterative_ruler.py
"""

import matplotlib.pyplot as plt

# =============================================================================
# DATA FROM iterative_ruler.tex
# =============================================================================
# X = iterations 1–5, Y = Avg. (RULER accuracy %)

ITERATIONS = [1, 2, 3, 4, 5]

# Random 2%: KV² and KV²+
KV2_RAND_2PCT = [17.3, 32.51, 36.91, 39.90, 40.30]
KV2_PLUS_RAND_2PCT = [18.09, 40.09, 43.68, 47.72, 49.61]

# Random 10%: KV² and KV²+
KV2_RAND_10PCT = [20.17, 35.28, 38.25, 39.25, 42.06]
KV2_PLUS_RAND_10PCT = [21.92, 41.06, 46.64, 48.79, 50.73]

# KeyDiff 2%: KV² and KV²+
KV2_KD_2PCT = [53.36, 54.66, 54.53, 55.51, 55.85]
KV2_PLUS_KD_2PCT = [51.54, 55.40, 57.76, 57.75, 59.73]

# KeyDiff 10%: KV² and KV²+
KV2_KD_10PCT = [62.77, 63.13, 62.37, 61.23, 59.59]
KV2_PLUS_KD_10PCT = [60.87, 62.03, 62.54, 62.11, 61.41]

FULL_SCORE = 95.61  # Full KV cache baseline (Llama-3.1-8B-Instruct)

# =============================================================================
# PLOT SETTINGS
# =============================================================================

FIGSIZE = (10, 6)
XLABEL = "Iterations"
YLABEL = "RULER Accuracy (%)"

COLOR_RAND_2PCT = "#4A90D9"
COLOR_RAND_10PCT = "#27AE60"
COLOR_KD_2PCT = "#D94A4A"
COLOR_KD_10PCT = "#E8922E"

# =============================================================================
# PLOTTING
# =============================================================================


def main():
    plt.figure(figsize=FIGSIZE)

    mk = dict(marker="o", markersize=8, linewidth=2)

    # Random 2% — blue
    plt.plot(ITERATIONS, KV2_PLUS_RAND_2PCT, label="KV²+ Random (2%)",
             color=COLOR_RAND_2PCT, linestyle="-", **mk)
    plt.plot(ITERATIONS, KV2_RAND_2PCT, label="KV² Random (2%)",
             color=COLOR_RAND_2PCT, linestyle=":", **mk)

    # Random 10% — green
    plt.plot(ITERATIONS, KV2_PLUS_RAND_10PCT, label="KV²+ Random (10%)",
             color=COLOR_RAND_10PCT, linestyle="-", **mk)
    plt.plot(ITERATIONS, KV2_RAND_10PCT, label="KV² Random (10%)",
             color=COLOR_RAND_10PCT, linestyle=":", **mk)

    # KeyDiff 2% — red
    plt.plot(ITERATIONS, KV2_PLUS_KD_2PCT, label="KV²+ KeyDiff (2%)",
             color=COLOR_KD_2PCT, linestyle="-", **mk)
    plt.plot(ITERATIONS, KV2_KD_2PCT, label="KV² KeyDiff (2%)",
             color=COLOR_KD_2PCT, linestyle=":", **mk)

    # KeyDiff 10% — orange
    plt.plot(ITERATIONS, KV2_PLUS_KD_10PCT, label="KV²+ KeyDiff (10%)",
             color=COLOR_KD_10PCT, linestyle="-", **mk)
    plt.plot(ITERATIONS, KV2_KD_10PCT, label="KV² KeyDiff (10%)",
             color=COLOR_KD_10PCT, linestyle=":", **mk)

    plt.xlabel(XLABEL, fontsize=12)
    plt.ylabel(YLABEL, fontsize=12)
    plt.title(
        "RULER-4k: Self-Refinement Steps (Llama-3.1-8B-Instruct, 2% KV Cache)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 5.5)
    plt.ylim(0, 70)
    plt.xticks(ITERATIONS)

    plt.tight_layout()
    plt.savefig("iterative_ruler.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: iterative_ruler.png")
    plt.show()


if __name__ == "__main__":
    main()
