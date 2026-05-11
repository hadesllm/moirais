# morie.fn — function file (hadesllm/morie)
"""Score norms — normative table from score distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_norms(
    scores: np.ndarray | pd.Series,
    *,
    percentiles: list[int] | None = None,
) -> dict:
    """Compute normative table from a score distribution.

    Parameters
    ----------
    scores : array-like
        Observed scores.
    percentiles : list[int] or None
        Percentile cut-points (default [5, 25, 50, 75, 95]).

    Returns
    -------
    dict
        Keys: 'n', 'mean', 'sd', 'min', 'max', 'percentiles' (dict).

    References
    ----------
    Angoff, W. H. (1971). Scales, norms, and equivalent scores. In
    R. L. Thorndike (Ed.), Educational Measurement (2nd ed.).
    """
    if percentiles is None:
        percentiles = [5, 25, 50, 75, 95]
    s = np.asarray(scores, dtype=np.float64).ravel()
    valid = s[~np.isnan(s)]
    pctile_vals = {int(p): float(np.percentile(valid, p)) for p in percentiles} if len(valid) > 0 else {}
    return {
        "n": len(valid),
        "mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "sd": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
        "min": float(np.min(valid)) if len(valid) > 0 else np.nan,
        "max": float(np.max(valid)) if len(valid) > 0 else np.nan,
        "percentiles": pctile_vals,
    }


def cheatsheet() -> str:
    return "score_norms({}) -> Score norms — normative table from score distribution."
