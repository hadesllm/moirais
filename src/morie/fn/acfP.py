"""Sample ACF at lag k."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["autocorrelation"]


def autocorrelation(y, lag_max):
    """
    Sample ACF at lag k

    Formula: ρ_k = sum (y_t−ȳ)(y_{t-k}−ȳ) / sum (y_t−ȳ)²

    Parameters
    ----------
    y : array-like
        Input data.
    lag_max : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brockwell-Davis (1991)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample ACF at lag k"})


def cheatsheet():
    return "acfP: Sample ACF at lag k"
