"""
Draft template to manually fill Needle-in-Haystack recall values.

For each method:
- one list per context length
- each list has recall values ordered by DEPTHS
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
})
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DEPTHS = [15, 25, 35, 45, 55, 65, 75, 85, 95]
CONTEXT_LENGTHS = list(range(10000, 130001, 10000))
OUTPUT_PATH = Path(__file__).with_name("needle_in_haystack.pdf")


# Fill values in DEPTHS order: [d15, d25, d35, d45, d55, d65, d75, d85, d95]
KEYDIFF = {
    "len_10000":  [0.6364, 0.3636, 0.1818, 0.3636, 0.6364, 0.3636, 1.0, 0.2727, 1.0],
    "len_20000":  [0.1818, 0.1818, 0.3636, 0.0, 0.6364, 0.2727, 0.1818, 0.3636, 0.6364],
    "len_30000":  [0.6364, 0.0, 0.3636, 0.0, 0.0, 0.1818, 0.0, 0.3636, 0.6364],
    "len_40000":  [0.0909, 0.2727, 0.0, 0.2727, 0.3636, 0.3636, 0.1818, 0.1818, 1.0],
    "len_50000":  [0.0909, 0.0, 0.1818, 0.0909, 0.3636, 0.0909, 0.3636, 0.3636, 0.6364],
    "len_60000":  [0.3636, 0.3636, 0.3636, 0.0909, 0.0909, 0.2727, 0.1818, 0.0, 0.6364],
    "len_70000":  [0.1818, 0.1818, 0.0, 0.1818, 0.0, 0.1818, 0.1818, 0.1818, 0.6364],
    "len_80000":  [0.1818, 0.0909, 0.1818, 0.0, 0.1818, 0.1818, 0.1818, 0.1818, 0.1818],
    "len_90000":  [0.0, 0.0, 0.1818, 0.1818, 0.1818, 0.0, 0.2727, 0.1818, 0.1818],
    "len_100000": [0.2727, 0.2727, 0.1818, 0.0, 0.0, 0.1818, 0.1818, 0.1818, 0.6364],
    "len_110000": [0.0, 0.1818, 0.6364, 0.1818, 0.2727, 0.3636, 0.0, 0.0, 0.3636],
    "len_120000": [0.2727, 0.1818, 0.1818, 0.1818, 0.1818, 0.1818, 0.0, 0.1818, 0.1818],
    "len_130000": [0.0909, 0.0, 0.6364, 0.0, 0.1818, 0.0, 0.0, 0.1818, 0.6364],
}

KVZIP = {
    "len_10000":  [0.6364, 0.6364, 0.6364, 0.1818, 0.0, 0.0, 1.0, 1.0, 1.0],
    "len_20000":  [1.0, 0.0, 0.6364, 1.0, 0.6364, 0.0, 1.0, 0.6364, 0.6364],
    "len_30000":  [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "len_40000":  [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    "len_50000":  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.6364, 0.6364],
    "len_60000":  [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    "len_70000":  [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "len_80000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6364, 0.0],
    "len_90000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_100000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "len_110000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_120000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1818, 0.0, 0.0],
    "len_130000": [0.1818, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6364, 0.0],
}

KVSQUARED = {
    "len_10000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_20000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_30000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_40000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_50000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_60000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_70000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7273, 1.0],
    "len_80000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7273, 1.0, 1.0],
    "len_90000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    "len_100000": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    "len_110000": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1818],
    "len_120000": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_130000": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

EXPECTED_ATTENTION = {
    "len_10000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_20000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_30000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_40000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_50000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_60000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_70000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_80000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_90000":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_100000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_110000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_120000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "len_130000": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


RECALL_DRAFT = {
    "KeyDiff": KEYDIFF,
    "KVzip": KVZIP,
    "Expected Attention": EXPECTED_ATTENTION,
    r"KV$^2$": KVSQUARED,
}


def _to_matrix(method_data: Dict[str, List[Optional[float]]]) -> np.ndarray:
    """Convert len_* lists into a [depth, context] matrix with NaN for missing values."""
    columns = []
    for context_length in CONTEXT_LENGTHS:
        key = f"len_{context_length}"
        if key not in method_data:
            raise KeyError(f"Missing key '{key}'.")
        values = method_data[key]
        if len(values) != len(DEPTHS):
            raise ValueError(
                f"'{key}' must contain {len(DEPTHS)} values, got {len(values)}."
            )
        columns.append([np.nan if v is None else float(v) for v in values])

    matrix = np.array(columns, dtype=float).T
    if np.any((~np.isnan(matrix)) & ((matrix < 0.0) | (matrix > 1.0))):
        raise ValueError("Recall values must be in [0.0, 1.0].")
    return matrix


def plot_heatmaps() -> Path:
    fig, axes_2d = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    fig.set_constrained_layout_pads(hspace=0.12)
    axes = axes_2d.flatten()

    # High recall = bright teal-green, low recall = vivid coral-red.
    cmap = LinearSegmentedColormap.from_list(
        "needle_style",
         [
            (0.0,  "#d93a54"),
            (0.1, "#eb6e5a"),
            (0.2,  "#f0a05a"),
            (0.3,  "#e8c84a"),
            # (0.4, "#9dd870"),
            (0.4,  "#48c78e"),
            (0.5,  "#2abb7f"),
            (1.0,  "#2abb7f"),
        ],
        N=256,
    )
    cmap.set_bad(color="#d9d9d9")

    images = []
    for idx, (ax, (method, method_data)) in enumerate(zip(axes, RECALL_DRAFT.items())):
        row, col = divmod(idx, 2)

        matrix = _to_matrix(method_data)
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(
            masked,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
            interpolation="nearest",
        )
        images.append(image)

        ax.set_title(method, fontsize=14)

        ax.set_xticks(range(len(CONTEXT_LENGTHS)))
        ax.set_xticklabels([str(v) for v in CONTEXT_LENGTHS], rotation=45, ha="right", fontsize=10)
        if row == 1:
            ax.set_xlabel("Context Length", fontsize=12)

        ax.set_yticks(range(len(DEPTHS)))
        ax.set_yticklabels([str(v) for v in DEPTHS], fontsize=10)
        if col == 0:
            ax.set_ylabel("Depth Percent", fontsize=12)

        # Draw thin white grid lines between cells
        ax.set_xticks(np.arange(-0.5, len(CONTEXT_LENGTHS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DEPTHS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # colorbar = fig.colorbar(images[0], ax=axes.tolist(), shrink=0.6, pad=0.02)
    # colorbar.set_label("Recall (ROUGE-2)", fontsize=12)
    fig.suptitle("NIAH - Llama-3.1-8B-Instruct", fontsize=14)

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    output_path = plot_heatmaps()
    print(f"Saved heatmap image to: {output_path}")


# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# output_dir: "./results"

# model: "meta-llama/Meta-Llama-3.1-8B-Instruct" # "meta-llama/Meta-Llama-3.1-8B-Instruct" or "Qwen/Qwen3-8B" or "Qwen/Qwen2.5-7B-Instruct-1M" or "google/gemma-3-12b"
# dataset: "needle_in_haystack"                                  # see DATASET_REGISTRY in evaluate_registry.py
# data_dir: ""                                # Subdirectory of the dataset (if applicable) else leave "null"

# press_name: "kvzip"                               # see PRESS_REGISTRY in evaluate_registry.py
# compression_ratio: 0.95                           # Compression ratio for the press (0.0 to 1.0)
# key_channel_compression_ratio: null               # For ThinKPress and ComposedPress (0.0 to 1.0)
# threshold: null                                   # For ThresholdPress

# fraction: 1.0                                     # Fraction of dataset to evaluate (0.0 to 1.0), for quick testing
# max_new_tokens: null                              # Maximum new tokens to generate (null = use dataset default)
# max_context_length: 100000                          # Maximum context length (null = use model maximum)
# query_aware: false                                # Whether to include question in context for query-aware compression
# needle_depth: [15, 25, 35, 45, 55, 65, 75, 85, 95]                                # Depth (int or list of ints) percentage of the needle in the haystack (0 to 100), only for needle_in_haystack dataset

# device: null  # Device to use (null = auto-detect, "cuda:0", "cpu", etc.)
# fp8: false    # Whether to use FP8 quantization (FineGrainedFP8Config() from transformers)

# # You can add any model kwargs here.
# model_kwargs:
#   attn_implementation: null  
#   dtype: "auto"
  

# Grahams Essays
# Needle - Remember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day. 
# Question - Question: Based on the content of the book, what is the best thing to do in San Francisco?
# Answer Prefix - Answer: The best thing to do in San Francisco is

# Example Outputs:
# Correct:
# eat a sandwich and sit in Dolores Park on a sunny day.

# Wrong:
# to visit the Exploratorium, a museum of science, art, and human perception.
# to visit the Golden Gate Bridge.
#  not explicitly mentioned in the provided content. The text appears to be a collection of essays and articles on various topics, including startup culture, programming, and language design. It does not contain any information about
#  to visit Dolores Park, as mentioned in the book.
#  to start a startup.
# to visit the startup scene, which is thriving in the city.