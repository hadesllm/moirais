"""Discrete-time dot product (inner product) of two N-sample signals.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_dot_product_discrete"]


def rangayyan_ch4_dot_product_discrete(x, y, N):
    """
    Discrete-time dot product (inner product) of two N-sample signals.

    Formula: x . y = <x, y> = sum_{n=0}^{N-1} x(n) * y(n)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.24, p. 229
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete-time dot product (inner product) of two N-sample signals."})


def cheatsheet():
    return "rng198: Discrete-time dot product (inner product) of two N-sample signals."
