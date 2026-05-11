# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Outlier detection. 'It's a trap!' -- Admiral Ackbar"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def detect_outliers(
    data: pd.DataFrame,
    *,
    col: str = "x",
    method: str = "iqr",
    threshold: float = 1.5,
) -> DescriptiveResult:
    """Detect outliers using IQR (default) or Z-score method."""
    _validate_df(data, col)
    x = data[col].dropna().to_numpy(dtype=float)
    if method == "iqr":
        q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        outlier_mask = (x < lower) | (x > upper)
    elif method == "zscore":
        z = np.abs((x - x.mean()) / x.std())
        outlier_mask = z > threshold
        lower, upper = x.mean() - threshold * x.std(), x.mean() + threshold * x.std()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")
    n_outliers = int(outlier_mask.sum())
    return DescriptiveResult(
        name=f"Outlier detection ({method})",
        value=n_outliers,
        extra={
            "n_total": len(x),
            "n_outliers": n_outliers,
            "pct_outliers": round(n_outliers / len(x) * 100, 1),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_values": x[outlier_mask].tolist()[:20],
        },
    )


ackbar = detect_outliers


def cheatsheet() -> str:
    return "detect_outliers({}) -> Outlier detection. 'It's a trap!' -- Admiral Ackbar"
