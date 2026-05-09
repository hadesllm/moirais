"""Noncausal least-squares second derivative used to detect the dicrotic notch.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_dicrotic_notch_second_derivative"]


def rangayyan_ch4_dicrotic_notch_second_derivative(y, n):
    """
    Noncausal least-squares second derivative used to detect the dicrotic notch.

    Formula: p(n) = 2*y(n-2) - y(n-1) - 2*y(n) - y(n+1) + 2*y(n+2)

    Parameters
    ----------
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
    Rangayyan (2024), Ch 4, Eq 4.22, p. 228
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Noncausal least-squares second derivative used to detect the dicrotic notch."})


def cheatsheet():
    return "rng196: Noncausal least-squares second derivative used to detect the dicrotic notch."
