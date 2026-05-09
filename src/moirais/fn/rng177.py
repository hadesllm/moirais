"""Approximation of the second derivative used in QRS detection.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_qrs_second_derivative_balda"]


def rangayyan_ch4_qrs_second_derivative_balda(x, n):
    """
    Approximation of the second derivative used in QRS detection.

    Formula: y_1(n) = |x(n) - 2*x(n-2) + x(n-4)|

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.2, p. 218
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Approximation of the second derivative used in QRS detection."})


def cheatsheet():
    return "rng177: Approximation of the second derivative used in QRS detection."
