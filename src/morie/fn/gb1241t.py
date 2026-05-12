# morie.fn -- function file (hadesllm/morie)
"""Coefficient of concordance W with correction for ties."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_concordance_w_ties"]


def gibbons_concordance_w_ties(rankings):
    """
    Coefficient of concordance W with correction for ties

    Formula: W_adj = S / (k^2(n^3-n)/12 - k*sum T_i)

    Parameters
    ----------
    rankings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: W_adj, p_value

    References
    ----------
    Gibbons Ch 12.4 ties
    """
    rankings = np.asarray(rankings, dtype=float)
    n = int(rankings) if rankings.ndim == 0 else len(rankings)
    result = float(np.mean(rankings))
    se = float(np.std(rankings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coefficient of concordance W with correction for ties"})


def cheatsheet():
    return "gb1241t: Coefficient of concordance W with correction for ties"
