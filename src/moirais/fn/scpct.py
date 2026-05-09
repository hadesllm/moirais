# moirais.fn — function file (hadesllm/moirais)
"""Percentile norm table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DescriptiveResult


def percentile_norms(
    scores: np.ndarray | pd.Series,
    *,
    percentiles: list[int] | None = None,
) -> DescriptiveResult:
    """Create a percentile norm table from raw scores.

    Parameters
    ----------
    scores : array-like
        Raw score vector (n,).
    percentiles : list[int], optional
        Percentile values to compute. Default: 1,5,10,25,50,75,90,95,99.

    Returns
    -------
    DescriptiveResult
        value=DataFrame with percentile and score_value columns.

    References
    ----------
    Angoff, W. H. (1984). Scales, Norms, and Equivalent Scores.
    ETS Research Report.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = s[~np.isnan(s)]

    if percentiles is None:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    vals = np.percentile(s, percentiles)
    df = pd.DataFrame(
        {
            "percentile": percentiles,
            "score_value": vals,
        }
    )

    return DescriptiveResult(
        name="Percentile norms",
        value=df,
        extra={"n": len(s), "mean": float(np.mean(s)), "sd": float(np.std(s, ddof=1))},
    )


pct_norms = percentile_norms


def cheatsheet() -> str:
    return "percentile_norms({}) -> Percentile norm table."
