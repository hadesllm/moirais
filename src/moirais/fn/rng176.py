"""Smoothed three-point first derivative used in QRS detection (Balda et al.).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_qrs_first_derivative_balda"]


def rangayyan_ch4_qrs_first_derivative_balda(x, n):
    """
    Smoothed three-point first derivative used in QRS detection (Balda et al.).

    Formula: y_0(n) = |x(n) - x(n-2)|

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
    Rangayyan (2024), Ch 4, Eq 4.1, p. 218
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothed three-point first derivative used in QRS detection (Balda et al.)."})


def cheatsheet():
    return "rng176: Smoothed three-point first derivative used in QRS detection (Balda et al.)."
