# morie.fn -- function file (hadesllm/morie)
"""Missing data pattern analysis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def missing_pattern(data: np.ndarray) -> DescriptiveResult:
    """Analyse missing-data patterns in a matrix.

    Parameters
    ----------
    data : (n, p) array (may contain np.nan)

    Returns
    -------
    DescriptiveResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n, p = data.shape

    miss_per_col = np.isnan(data).sum(axis=0)
    miss_per_row = np.isnan(data).sum(axis=1)
    total_missing = int(np.isnan(data).sum())
    pct_missing = total_missing / (n * p) * 100

    patterns = {}
    for i in range(n):
        pat = tuple(np.isnan(data[i]).astype(int))
        patterns[pat] = patterns.get(pat, 0) + 1

    complete_rows = int(np.sum(miss_per_row == 0))

    return DescriptiveResult(
        name="missing_pattern",
        value=float(pct_missing),
        extra={
            "total_missing": total_missing,
            "pct_missing": float(pct_missing),
            "missing_per_col": miss_per_col.tolist(),
            "complete_rows": complete_rows,
            "n_patterns": len(patterns),
            "n": n,
            "p": p,
        },
    )


mispt = missing_pattern


def cheatsheet() -> str:
    return "missing_pattern({}) -> Missing data pattern analysis."
