"""
Draft template to manually fill Needle-in-Haystack recall values.

For each method:
- one list per context length
- each list has recall values ordered by DEPTHS
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DEPTHS = [15, 25, 35, 45, 55, 65, 75, 85, 95]
CONTEXT_LENGTHS = list(range(10000, 130001, 10000))
OUTPUT_PATH = Path(__file__).with_name("needle_recall_heatmaps.png")


# Fill values in DEPTHS order: [d15, d25, d35, d45, d55, d65, d75, d85, d95]
KEYDIFF = {
    "len_10000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],
    "len_20000":  [1.0, 0.5714, 1.0, 0.5714, 1.0, 0.6364, 1.0, 1.0, 1.0],
    "len_30000":  [1.0, 0.4545, 0.5714, 0.2759, 0.2143, 1.0, 0.1786, 1.0, 1.0],
    "len_40000":  [0.5714, 0.7368, 0.2143, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_50000":  [0.5556, 0.3684, 0.6364, 0.4545, 1.0, 0.4231, 1.0, 1.0, 1.0],
    "len_60000":  [1.0, 1.0, 1.0, 0.25, 0.1875, 1.0, 1.0, 0.2727, 1.0],
    "len_70000":  [1.0, 1.0, 0.875, 1.0, 0.1579, 1.0, 1.0, 1.0, 1.0],
    "len_80000":  [1.0, 0.2857, 1.0, 0.3333, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_90000":  [0.2667, 0.2667, 1.0, 1.0, 1.0, 0.4444, 0.3333, 1.0, 1.0],
    "len_100000": [0.8571, 0.8571, 1.0, 0.3333, 0.3333, 1.0, 1.0, 1.0, 1.0],
    "len_110000": [0.4444, 0.8571, 1.0, 0.6, 0.6667, 1.0, 0.75, 0.25, 1.0],
    "len_120000": [0.8571, 0.8571, 0.7143, 1.0, 1.0, 1.0, 0.3333, 1.0, 1.0],
    "len_130000": [0.75, 0.375, 1.0, 0.4, 1.0, 0.2, 0.2857, 1.0, 1.0],
}

KVZIP = {
    "len_10000":  [1.0, 1.0, 1.0, 1.0, 1.0, 0.2857, 1.0, 1.0, 1.0],
    "len_20000":  [1.0, 1.0, 1.0, 1.0, 1.0, 0.1818, 1.0, 1.0, 1.0],
    "len_30000":  [1.0, 0.3333, 0.3333, 1.0, 0.3333, 1.0, 0.3333, 0.3333, 0.3333],
    "len_40000":  [0.3571, 0.1667, 0.3571, 1.0, 1.0, 0.3333, 1.0, 0.3333, 1.0],
    "len_50000":  [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.1818, 0.3333, 1.0, 1.0],
    "len_60000":  [0.3333, 1.0, 1.0, 0.3333, 1.0, 1.0, 0.3333, 0.3333, 1.0],
    "len_70000":  [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 1.0, 0.3333, 0.3333, 1.0],
    "len_80000":  [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 1.0, 0.3333],
    "len_90000":  [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333],
    "len_100000": [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 1.0, 0.3333, 0.3333],
    "len_110000": [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333],
    "len_120000": [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 1.0, 0.3333, 0.3333],
    "len_130000": [1.0, 0.4, 0.3333, 0.4, 0.1818, 1.0, 0.2667, 1.0, 0.4],
}

KVSQUARED = {
    "len_10000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_20000":  [1.0, 1.0, 0.8462, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_30000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_40000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_50000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_60000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_70000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_80000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_90000":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2632, 1.0, 1.0],
    "len_100000": [1.0, 1.0, 1.0, 1.0, 0.3333, 1.0, 1.0, 1.0, 1.0],
    "len_110000": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_120000": [1.0, 1.0, 1.0, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0],
    "len_130000": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

EXPECTED_ATTENTION = {
    "len_10000":  [0.1818, 0.1818, 0.1818, 0.1818, 0.1818, 0.1818, 0.1818, 0.1818, 0.1935],
    "len_20000":  [0.1818, 0.1818, 0.1818, 0.1818, 0.1818, 0.1765, 0.1818, 0.1765, 0.1818],
    "len_30000":  [0.4444, 0.1852, 0.5, 0.5, 0.2, 0.3333, 0.5, 0.5, 0.5],
    "len_40000":  [0.2083, 0.1923, 0.1923, 0.2143, 0.1923, 0.1765, 0.2143, 0.1923, 0.4667],
    "len_50000":  [0.2069, 0.3, 0.3333, 0.3333, 0.3, 0.3333, 0.2273, 0.3333, 0.3],
    "len_60000":  [0.2381, 0.2381, 0.2667, 0.2667, 0.3333, 0.4211, 0.2381, 0.2667, 0.3333],
    "len_70000":  [0.3571, 0.3571, 0.3889, 0.2857, 0.2857, 0.3571, 0.3571, 0.2174, 0.3],
    "len_80000":  [0.2593, 0.3571, 0.2593, 0.3571, 0.3, 0.3571, 0.3571, 0.2593, 0.2857],
    "len_90000":  [0.15, 0.2143, 0.15, 0.15, 0.15, 0.2593, 0.15, 0.15, 0.2143],
    "len_100000": [0.3333, 0.3846, 0.3846, 0.2941, 0.3333, 0.2941, 0.3846, 0.2941, 0.2941],
    "len_110000": [0.3333, 0.3333, 0.3333, 0.3846, 0.3846, 0.3846, 0.3846, 0.3846, 0.3846],
    "len_120000": [0.125, 0.5, 0.3125, 0.3125, 0.5, 0.5, 0.3125, 0.3125, 0.3125],
    "len_130000": [0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.6, 0.6, 0.3333, 0.2857],
}


RECALL_DRAFT = {
    "KeyDiff": KEYDIFF,
    "KVzip": KVZIP,
    "Expected Attention": EXPECTED_ATTENTION,
    "KV²": KVSQUARED,
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
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes = axes.flatten()

    # High recall = bright green, low recall = red. Everything below 0.2 is already red.
    cmap = LinearSegmentedColormap.from_list(
        "needle_style",
        [
            (0.0,  "#d93a54"),
            (0.2,  "#d93a54"),
            (0.35, "#eb6e5a"),
            (0.5,  "#f0a05a"),
            (0.6,  "#e8c84a"),
            (0.75, "#9dd870"),
            (0.9,  "#48c78e"),
            (1.0,  "#2abb7f"),
        ],
        N=256,
    )
    cmap.set_bad(color="#d9d9d9")

    images = []
    for ax, (method, method_data) in zip(axes, RECALL_DRAFT.items()):
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

        ax.set_title(method, fontsize=18)
        ax.set_xlabel("Context Length", fontsize=12)
        ax.set_ylabel("Depth Percent", fontsize=12)

        ax.set_xticks(range(len(CONTEXT_LENGTHS)))
        ax.set_xticklabels([str(v) for v in CONTEXT_LENGTHS], rotation=45, ha="right")
        ax.set_yticks(range(len(DEPTHS)))
        ax.set_yticklabels([str(v) for v in DEPTHS])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(images[0], ax=axes.tolist(), shrink=0.9, pad=0.02)
    colorbar.set_label("Recall", fontsize=12)
    fig.suptitle("Needle-in-Haystack: Llama-3.1-8B, 5% KV size", fontsize=22, fontweight="bold")

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
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