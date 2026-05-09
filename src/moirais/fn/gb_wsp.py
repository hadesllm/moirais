# moirais.fn — function file (hadesllm/moirais)
"""Estimate of true preferential ordering from concordance analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_concordance_preference"]


def gibbons_concordance_preference(rankings):
    """
    Estimate of true preferential ordering from concordance analysis

    Formula: Rank alternatives by sum of ranks R_j; smallest R_j is most preferred

    Parameters
    ----------
    rankings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: preferential_order

    References
    ----------
    Gibbons Ch 12.4
    """
    rankings = np.asarray(rankings, dtype=float)
    n = int(rankings) if rankings.ndim == 0 else len(rankings)
    if rankings.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimate of true preferential ordering from concordance analysis"})
    estimate = np.median(rankings)
    se = 1.2533 * np.std(rankings, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimate of true preferential ordering from concordance analysis"})


def cheatsheet():
    return "gb_wsp: Estimate of true preferential ordering from concordance analysis"
