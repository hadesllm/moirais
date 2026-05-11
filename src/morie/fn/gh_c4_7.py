# morie.fn — function file (hadesllm/morie)
"""Predictive distribution of DP: Polya urn for new observation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_dp_pred"]


def ghosal_dp_pred(x):
    """
    Predictive distribution of DP: Polya urn for new observation

    Formula: X_{n+1}|X_1..X_n ~ (alpha*G0 + sum delta_{X_i})/(alpha+n)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 4 §4.1.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Predictive distribution of DP: Polya urn for new observation"})


def cheatsheet():
    return "gh_c4_7: Predictive distribution of DP: Polya urn for new observation"
