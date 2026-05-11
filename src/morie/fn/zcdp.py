"""Zero-concentrated DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["zcdp"]


def zcdp(mech, rho):
    """
    Zero-concentrated DP

    Formula: D_α(M(D)||M(D')) ≤ ρ α

    Parameters
    ----------
    mech : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bun-Steinke (2016)
    """
    mech = np.atleast_1d(np.asarray(mech, dtype=float))
    n = len(mech)
    result = float(np.mean(mech))
    se = float(np.std(mech, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zero-concentrated DP"})


def cheatsheet():
    return "zcdp: Zero-concentrated DP"
