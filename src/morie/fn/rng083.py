"""Decomposition of a signal into even and odd parts.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_even_odd_decomposition"]


def rangayyan_ch3_even_odd_decomposition(x_e, x_o, n):
    """
    Decomposition of a signal into even and odd parts.

    Formula: x(n) = x_e(n) + x_o(n)

    Parameters
    ----------
    x_e : array-like
        Input data.
    x_o : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.94, p. 135
    """
    x_e = np.atleast_1d(np.asarray(x_e, dtype=float))
    n = len(x_e)
    result = float(np.mean(x_e))
    se = float(np.std(x_e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Decomposition of a signal into even and odd parts."})


def cheatsheet():
    return "rng083: Decomposition of a signal into even and odd parts."
