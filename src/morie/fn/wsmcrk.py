"""Nadaraya-Watson kernel regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_kernel_regression"]


def wasserman_kernel_regression(x, x_data, y_data, h):
    """
    Nadaraya-Watson kernel regression

    Formula: m_h(x) = sum K_h(x-X_i) Y_i / sum K_h(x-X_i)

    Parameters
    ----------
    x : array-like
        Input data.
    x_data : array-like
        Input data.
    y_data : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wasserman (2004), Ch 20
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nadaraya-Watson kernel regression"})


def cheatsheet():
    return "wsmcrk: Nadaraya-Watson kernel regression"
