# morie.fn -- function file (hadesllm/morie)
"""Score cut-offs (tercile, quartile, clinical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ._richresult import RichResult


def score_cutoffs(
    scores: np.ndarray | pd.Series,
    *,
    method: str = "tercile",
) -> dict:
    """Compute cut scores from score distribution.

    Parameters
    ----------
    scores : array-like
        Observed scores.
    method : str
        One of 'tercile', 'quartile', 'median', 'mean_sd'.

    Returns
    -------
    dict
        Keys depend on method. Always includes 'method' and 'cuts'.

    References
    ----------
    Jacobson, N. S. & Truax, P. (1991). Clinical significance: A
    statistical approach. Journal of Consulting and Clinical Psychology,
    59(1), 12-19.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    valid = s[~np.isnan(s)]
    if len(valid) == 0:
        return RichResult(payload={"method": method, "cuts": [], "n": 0})

    if method == "tercile":
        cuts = [float(np.percentile(valid, p)) for p in [33.33, 66.67]]
        labels = ["low", "medium", "high"]
    elif method == "quartile":
        cuts = [float(np.percentile(valid, p)) for p in [25, 50, 75]]
        labels = ["Q1", "Q2", "Q3", "Q4"]
    elif method == "median":
        cuts = [float(np.median(valid))]
        labels = ["below_median", "above_median"]
    elif method == "mean_sd":
        m, sd = float(np.mean(valid)), float(np.std(valid, ddof=1))
        cuts = [m - sd, m, m + sd]
        labels = ["low", "below_avg", "above_avg", "high"]
    else:
        raise ValueError(f"Unknown method: {method}. Use tercile/quartile/median/mean_sd.")

    return RichResult(payload={"method": method, "cuts": cuts, "labels": labels, "n": len(valid)})


def cheatsheet() -> str:
    return "score_cutoffs({}) -> Score cut-offs (tercile, quartile, clinical)."
