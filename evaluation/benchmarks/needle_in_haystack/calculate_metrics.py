# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from rouge import Rouge

scorer = Rouge()


GROUND_TRUTH = "eat a sandwich and sit in Dolores Park on a sunny day."


def calculate_metrics(df: pd.DataFrame) -> list[dict]:
    scores = []
    for index, row in df.iterrows():
        score = scorer.get_scores(row["predicted_answer"].strip(), GROUND_TRUTH)[0]
        scores.append(score)
    return scores
