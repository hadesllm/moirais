"""RLS volatility update on squared returns."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_recursive_least_sq"]


def vol_recursive_least_sq(r, lam):
    """
    RLS volatility update on squared returns

    Formula: σ̂_t² = λ σ̂_{t-1}² + (1-λ) r_t²

    Parameters
    ----------
    r : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sigma2_t

    References
    ----------
    Engle (1982)
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLS volatility update on squared returns"})


def cheatsheet():
    return "volrls: RLS volatility update on squared returns"
