# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np

# Data - each method has values for each chunk size
chunk_sizes = ["0.5k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k"]

# Compute times for each method (use "OOM" for out-of-memory cases)
method_1_times  = [138, 131, 116, 121, 132, "OOM", "OOM", "OOM", "OOM"]
method_2_times = [108, 80, 72, 70, 73, 80, "OOM", "OOM", "OOM"]  
method_3_times = [98, 60, 41, 32, 27, 25, 25, 25, 26]
method_4_times = [0.0, 0.0, 0.0, 43, 34, 30, 30, 31, 35]
method_5_times = [0.0, 0.0, 0.0, 77, 54, 47, 49, 55, 78]

def process_data(data):
    """Convert data list, replacing 'OOM' with 0 and tracking OOM positions."""
    numeric_data = []
    oom_indices = []
    for i, val in enumerate(data):
        if val == "OOM":
            numeric_data.append(0)
            oom_indices.append(i)
        else:
            numeric_data.append(val)
    return numeric_data, oom_indices

# Define colors - all transparent
colors_fill = [
    (0.29, 0.56, 0.85, 0.5),      # Blue matching #4A90D9 - method 1
    (0.91, 0.30, 0.24, 0.5),      # Red matching #E74C3C - method 2
    (0.13, 0.55, 0.13, 0.5),      # Forest green matching #228B22 - method 4
    (0.0, 0.25, 0.0, 0.5),        # Dark green matching #004000 - method 5
    (0.15, 0.68, 0.38, 0.5),      # Green matching #27AE60 - method 3
]
colors_edge = [
    "#4A90D9",  # Blue edge - method 1
    "#E74C3C",  # Red edge - method 2
    "#228B22",  # Forest green edge - method 4 (back)
    "#004000",  # Very dark green edge - method 5 (back)
    "#27AE60",  # Green edge - method 3 (front)
]
# Reordered: methods 4 & 5 before method 3 so they appear behind
method_names = ["kvzip", "kvsquared_keydiff_0.5", "kvsquared_keydiff_2_iter", "kvsquared_keydiff_5_iter", "kvsquared_keydiff_0.02"]

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Bar positioning
x = np.arange(len(chunk_sizes))
bar_width = 0.25

# Process all data
method_1_data, method_1_oom = process_data(method_1_times)
method_2_data, method_2_oom = process_data(method_2_times)
method_3_data, _ = process_data(method_3_times)
method_4_data, _ = process_data(method_4_times)
method_5_data, _ = process_data(method_5_times)

# Calculate the extra heights for stacking:
# - Method 4 extra: above method 3
# - Method 5 extra: above method 4 (or above method 3 if method 4 has no data)
method_4_extra = []
method_5_extra = []
method_5_bottom = []  # Where method 5 bar starts
for j in range(len(chunk_sizes)):
    if method_4_data[j] > 0:
        method_4_extra.append(method_4_data[j] - method_3_data[j])
    else:
        method_4_extra.append(0)
    
    if method_5_data[j] > 0:
        if method_4_data[j] > 0:
            # Method 5 stacks on top of method 4
            method_5_extra.append(method_5_data[j] - method_4_data[j])
            method_5_bottom.append(method_4_data[j])
        else:
            # Method 4 has no data, method 5 stacks on top of method 3
            method_5_extra.append(method_5_data[j] - method_3_data[j])
            method_5_bottom.append(method_3_data[j])
    else:
        method_5_extra.append(0)
        method_5_bottom.append(method_3_data[j])

# Draw methods 1 and 2 (blue and red) - standalone bars
for i, (data, fill_color, edge_color, name, oom_indices, oom_y) in enumerate([
    (method_1_data, colors_fill[0], colors_edge[0], method_names[0], method_1_oom, 130),
    (method_2_data, colors_fill[1], colors_edge[1], method_names[1], method_2_oom, 90),
]):
    offset = -bar_width if i == 0 else 0
    bars = ax.bar(x + offset, data, bar_width, color=fill_color, edgecolor=edge_color, linewidth=1.5, label=name)
    
    # OOM annotations
    bar_centers = x + offset
    for oom_idx in oom_indices:
        ax.text(bar_centers[oom_idx], oom_y, "OOM", ha="center", va="center", fontsize=7, fontweight="bold", color=edge_color)
    
    # Line
    non_zero_mask = [v > 0 for v in data]
    line_x = [bar_centers[j] for j in range(len(data)) if non_zero_mask[j]]
    line_y = [data[j] for j in range(len(data)) if non_zero_mask[j]]
    if len(line_x) > 1:
        ax.plot(line_x, line_y, color=edge_color, linewidth=2, marker='o', markersize=5, zorder=5)

# Draw green stacked bars: method 3 as base, methods 4 & 5 as extra on top
green_offset = bar_width
bar_centers = x + green_offset

# Draw method 3 (base - light green)
bars_3 = ax.bar(bar_centers, method_3_data, bar_width, color=colors_fill[4], edgecolor=colors_edge[4], linewidth=1.5, label=method_names[4])

# Draw method 4 extra (stacked on top of method 3 - forest green)
bars_4 = ax.bar(bar_centers, method_4_extra, bar_width, bottom=method_3_data, color=colors_fill[2], edgecolor=colors_edge[2], linewidth=1.5, label=method_names[2])

# Draw method 5 extra (stacked on top of method 4, or method 3 if no method 4 data - dark green)
bars_5 = ax.bar(bar_centers, method_5_extra, bar_width, bottom=method_5_bottom, color=colors_fill[3], edgecolor=colors_edge[3], linewidth=1.5, label=method_names[3])

# Draw lines for each green method
for data, edge_color in [
    (method_3_data, colors_edge[4]),
    (method_4_data, colors_edge[2]),
    (method_5_data, colors_edge[3]),
]:
    non_zero_mask = [v > 0 for v in data]
    line_x = [bar_centers[j] for j in range(len(data)) if non_zero_mask[j]]
    line_y = [data[j] for j in range(len(data)) if non_zero_mask[j]]
    if len(line_x) > 1:
        ax.plot(line_x, line_y, color=edge_color, linewidth=2, marker='o', markersize=5, zorder=5)

# Add horizontal dashed line at 20s
ax.axhline(y=20, color="black", linestyle="--", linewidth=1.5)

# Customize axes
ax.set_xticks(x)
ax.set_xticklabels(chunk_sizes)
ax.set_xlabel("Repeat chunk size", fontsize=12)
ax.set_ylabel("Compute time (s)", fontsize=12)

# Set y-axis limits and ticks
ax.set_ylim(0, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100, 120, 140, 160])

# Add grid lines (horizontal only, dashed)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add legend
ax.legend(loc="upper center", fontsize=10, ncol=3)

plt.tight_layout()

# Save figure
plt.savefig("chunk_size_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: chunk_size_comparison.png and chunk_size_comparison.pdf")

plt.show()
