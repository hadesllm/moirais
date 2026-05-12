# morie.fn -- function file (hadesllm/morie)
"""Correlation pattern finder. 'I need a pilot!' -- Finn"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import DescriptiveResult


def find_patterns(
    data: pd.DataFrame,
    *,
    threshold: float = 0.3,
    method: str = "pearson",
) -> DescriptiveResult:
    """Scan all numeric column pairs for significant correlations."""
    numeric = data.select_dtypes(include=[np.number])
    cols = numeric.columns.tolist()
    pairs = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            valid = numeric[[c1, c2]].dropna()
            if len(valid) < 3:
                continue
            if method == "spearman":
                r, p = stats.spearmanr(valid[c1], valid[c2])
            elif method == "kendall":
                r, p = stats.kendalltau(valid[c1], valid[c2])
            else:
                r, p = stats.pearsonr(valid[c1], valid[c2])
            if abs(r) >= threshold:
                pairs.append({"col1": c1, "col2": c2, "r": round(float(r), 4), "p": float(p), "n": len(valid)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    return DescriptiveResult(
        name="Correlation scan",
        value=len(pairs),
        extra={"pairs": pairs, "method": method, "threshold": threshold},
    )


finn = find_patterns


def cheatsheet() -> str:
    return "find_patterns({}) -> Correlation pattern finder. 'I need a pilot!' -- Finn"
