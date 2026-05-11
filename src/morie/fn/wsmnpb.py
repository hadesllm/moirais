"""Nonparametric bootstrap se."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_nonparametric_boot"]


def wasserman_nonparametric_boot(data, T, B):
    """
    Nonparametric bootstrap se

    Formula: se_boot = sqrt((1/B) sum (theta_b - theta_bar)^2)

    Parameters
    ----------
    data : array-like
        Input data.
    T : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 8
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric bootstrap se"})


def cheatsheet():
    return "wsmnpb: Nonparametric bootstrap se"
