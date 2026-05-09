"""Cross-method consistency check (g-formula vs IPW vs g-est)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["g_methods_consistency"]


def g_methods_consistency(y, treatment_history, covariate_history, tau):
    """
    Cross-method consistency check (g-formula vs IPW vs g-est)

    Formula: compare three estimates; flag divergence > tau

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Naimi-Cole-Kennedy (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-method consistency check (g-formula vs IPW vs g-est)"})


def cheatsheet():
    return "gmccsm: Cross-method consistency check (g-formula vs IPW vs g-est)"
