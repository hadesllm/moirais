# moirais.fn — function file (hadesllm/moirais)
"""Score equating between forms."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_equate(
    scores1: np.ndarray | pd.Series,
    scores2: np.ndarray | pd.Series,
    *,
    method: str = "equipercentile",
) -> dict:
    """Equate scores from two test forms.

    Parameters
    ----------
    scores1 : array-like
        Scores from Form 1 (reference).
    scores2 : array-like
        Scores from Form 2 (to be equated).
    method : str
        'linear' or 'equipercentile' (default).

    Returns
    -------
    dict
        Keys: 'method', 'form1_stats', 'form2_stats', 'concordance'.
        concordance maps Form 2 unique scores to equated Form 1 equivalents.

    References
    ----------
    Kolen, M. J. & Brennan, R. L. (2014). Test Equating, Scaling, and
    Linking (3rd ed.). Springer.
    """
    s1 = np.asarray(scores1, dtype=np.float64).ravel()
    s2 = np.asarray(scores2, dtype=np.float64).ravel()
    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]

    stats1 = {"mean": float(np.mean(s1)), "sd": float(np.std(s1, ddof=1)), "n": len(s1)}
    stats2 = {"mean": float(np.mean(s2)), "sd": float(np.std(s2, ddof=1)), "n": len(s2)}

    if method == "linear":
        # Linear: Y* = (sd1/sd2)(X - mean2) + mean1
        ratio = stats1["sd"] / stats2["sd"] if stats2["sd"] > 1e-15 else 1.0
        unique2 = np.sort(np.unique(s2))
        concordance = {float(x): float(ratio * (x - stats2["mean"]) + stats1["mean"]) for x in unique2}
    elif method == "equipercentile":
        unique2 = np.sort(np.unique(s2))
        concordance = {}
        for x in unique2:
            # Percentile rank of x in Form 2
            pr = float(np.mean(s2 <= x) * 100)
            # Find equivalent in Form 1 at same percentile
            equiv = float(np.percentile(s1, min(pr, 100)))
            concordance[float(x)] = equiv
    else:
        raise ValueError(f"Unknown method: {method}. Use linear/equipercentile.")

    return {
        "method": method,
        "form1_stats": stats1,
        "form2_stats": stats2,
        "concordance": concordance,
    }


def cheatsheet() -> str:
    return "score_equate({}) -> Score equating between forms."
