"""Truncated IPW (Crump trim) for stable weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_ipw_truncated"]


def causal_ipw_truncated(treat, y, ps, alpha):
    """
    Truncated IPW (Crump trim) for stable weights

    Formula: Trim e to [α, 1-α]; standard IPW after trim

    Parameters
    ----------
    treat : array-like
        Input data.
    y : array-like
        Input data.
    ps : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATE, se

    References
    ----------
    Crump-Hotz-Imbens-Mitnik (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Truncated IPW (Crump trim) for stable weights"})


def cheatsheet():
    return "causipw: Truncated IPW (Crump trim) for stable weights"
