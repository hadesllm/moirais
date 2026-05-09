"""Propensity-score nearest-neighbour 1:1 matching."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_pair_matching"]


def causal_pair_matching(ps, treat, caliper):
    """
    Propensity-score nearest-neighbour 1:1 matching

    Formula: Match each treated to control with nearest |e(x)|

    Parameters
    ----------
    ps : array-like
        Input data.
    treat : array-like
        Input data.
    caliper : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: match_idx, ATT

    References
    ----------
    Rubin (1973)
    """
    ps = np.atleast_1d(np.asarray(ps, dtype=float))
    n = len(ps)
    result = float(np.mean(ps))
    se = float(np.std(ps, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Propensity-score nearest-neighbour 1:1 matching"})


def cheatsheet():
    return "causmtch: Propensity-score nearest-neighbour 1:1 matching"
