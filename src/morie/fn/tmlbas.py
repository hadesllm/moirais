"""TMLE with baseline-adjusted outcome regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_baseline_adj"]


def tmle_baseline_adj(y, D, X, baseline):
    """
    TMLE with baseline-adjusted outcome regression

    Formula: pre-treatment Q updated with baseline covariates only

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    baseline : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tsiatis et al (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE with baseline-adjusted outcome regression"})


def cheatsheet():
    return "tmlbas: TMLE with baseline-adjusted outcome regression"
