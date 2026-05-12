# morie.fn -- function file (hadesllm/morie)
"""Missing data pattern analysis."""

import numpy as np

from ._containers import DescriptiveResult


def missing_pattern(X):
    """
    Analyze and classify missing data patterns.

    :param X: (n, p) data matrix with NaN as missing.
    :return: DescriptiveResult with pattern summary, monotone check, % missing.

    References
    ----------
    Little RJA, Rubin DB (2019). Statistical Analysis with Missing Data.
    3rd ed. Wiley.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    miss = np.isnan(X)
    pct_missing = miss.mean(axis=0)
    n_complete = int((~miss).all(axis=1).sum())
    patterns = {}
    for i in range(n):
        pat = tuple(miss[i].astype(int))
        patterns[pat] = patterns.get(pat, 0) + 1

    sorted_cols = np.argsort(-pct_missing)
    monotone = True
    for i in range(n):
        first_obs = True
        for j in sorted_cols:
            if miss[i, j]:
                first_obs = False
            elif not first_obs:
                monotone = False
                break
        if not monotone:
            break

    return DescriptiveResult(
        name="missing_pattern",
        value=float(miss.mean()),
        extra={
            "pct_missing_overall": float(miss.mean()),
            "pct_missing_per_col": pct_missing.tolist(),
            "n_patterns": len(patterns),
            "n_complete_cases": n_complete,
            "is_monotone": monotone,
            "n": n,
            "p": p,
        },
    )


def cheatsheet() -> str:
    return "missing_pattern({}) -> Missing data pattern analysis."
