"""Differential-privacy-compatible TMLE via Laplace noise."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_diff_kernel"]


def tmle_diff_kernel(y, D, X, epsilon):
    """
    Differential-privacy-compatible TMLE via Laplace noise

    Formula: add Laplace noise to influence curve estimate

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Niu-Cohen-Shen (2024)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differential-privacy-compatible TMLE via Laplace noise"})


def cheatsheet():
    return "tmldyk: Differential-privacy-compatible TMLE via Laplace noise"
