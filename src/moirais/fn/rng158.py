"""Steepest-descent update rule for the tap-weight vector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_steepest_descent"]


def rangayyan_ch3_lms_steepest_descent(w, mu, n):
    """
    Steepest-descent update rule for the tap-weight vector.

    Formula: w(n+1) = w(n) - mu * grad(e^2(n))

    Parameters
    ----------
    w : array-like
        Input data.
    mu : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.201, p. 184
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Steepest-descent update rule for the tap-weight vector."})


def cheatsheet():
    return "rng158: Steepest-descent update rule for the tap-weight vector."
