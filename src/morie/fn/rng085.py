"""Synchronized sum across M observations to form ensemble averaging.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_synchronized_averaging_sum"]


def rangayyan_ch3_synchronized_averaging_sum(y_k, x_k, eta_k, n, M):
    """
    Synchronized sum across M observations to form ensemble averaging.

    Formula: sum_{k=1}^{M} y_k(n) = sum_{k=1}^{M} x_k(n) + sum_{k=1}^{M} eta_k(n)

    Parameters
    ----------
    y_k : array-like
        Input data.
    x_k : array-like
        Input data.
    eta_k : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.96, p. 135
    """
    y_k = np.atleast_1d(np.asarray(y_k, dtype=float))
    n = len(y_k)
    result = float(np.mean(y_k))
    se = float(np.std(y_k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Synchronized sum across M observations to form ensemble averaging."})


def cheatsheet():
    return "rng085: Synchronized sum across M observations to form ensemble averaging."
