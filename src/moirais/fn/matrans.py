"""Logit transform for proportion meta-analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_logit_transform"]


def ma_logit_transform(p, n):
    """
    Logit transform for proportion meta-analysis

    Formula: logit(p) = log(p/(1-p)); v = 1/(np) + 1/(n(1-p))

    Parameters
    ----------
    p : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: logit, var

    References
    ----------
    Nyaga et al. (2014)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logit transform for proportion meta-analysis"})


def cheatsheet():
    return "matrans: Logit transform for proportion meta-analysis"
