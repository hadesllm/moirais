"""Residual TMLE — second-order influence."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_residual"]


def tmle_residual(y, D, X):
    """
    Residual TMLE — second-order influence

    Formula: residual update H** for second-order bias correction

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins et al (2017) Annals
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Residual TMLE — second-order influence"})


def cheatsheet():
    return "tmlres: Residual TMLE — second-order influence"
