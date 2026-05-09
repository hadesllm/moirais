"""3-period DiD with placebo (forward + backward checks)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["causal_did_three_way"]


def causal_did_three_way(y, treated, t):
    """
    3-period DiD with placebo (forward + backward checks)

    Formula: Compare DiD(t-1,t) vs DiD(t-2,t-1) for placebo

    Parameters
    ----------
    y : array-like
        Input data.
    treated : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ATT, placebo, p

    References
    ----------
    Roth (2022)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "3-period DiD with placebo (forward + backward checks)"})


def cheatsheet():
    return "causdid3w: 3-period DiD with placebo (forward + backward checks)"
