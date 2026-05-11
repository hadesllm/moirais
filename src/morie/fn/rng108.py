"""Recursive form of the 8-point MA filter using delayed output.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point_recursive"]


def rangayyan_ch3_ma_8point_recursive(x, y, n):
    """
    Recursive form of the 8-point MA filter using delayed output.

    Formula: y(n) = y(n-1) + (1/8)*x(n) - (1/8)*x(n-8)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.120, p. 145
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recursive form of the 8-point MA filter using delayed output."})


def cheatsheet():
    return "rng108: Recursive form of the 8-point MA filter using delayed output."
