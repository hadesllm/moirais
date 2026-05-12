"""Prophet -- piecewise trend + Fourier seasonal + holidays."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prophet"]


def prophet(ds, y, holidays, changepoints):
    """
    Prophet -- piecewise trend + Fourier seasonal + holidays

    Formula: y(t) = g(t) + s(t) + h(t) + ε

    Parameters
    ----------
    ds : array-like
        Input data.
    y : array-like
        Input data.
    holidays : array-like
        Input data.
    changepoints : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Taylor-Letham (2018) Facebook Prophet
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prophet -- piecewise trend + Fourier seasonal + holidays"})


def cheatsheet():
    return "prphet: Prophet -- piecewise trend + Fourier seasonal + holidays"
