"""Adaptive FIR filter output in LMS framework using reference input r(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_filter_output"]


def rangayyan_ch3_lms_filter_output(r, w_k, n, M):
    """
    Adaptive FIR filter output in LMS framework using reference input r(n).

    Formula: y(n) = sum_{k=0}^{M-1} w_k * r(n - k)

    Parameters
    ----------
    r : array-like
        Input data.
    w_k : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.195, p. 183
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    result = float(np.mean(r))
    se = float(np.std(r, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive FIR filter output in LMS framework using reference input r(n)."})


def cheatsheet():
    return "rng155: Adaptive FIR filter output in LMS framework using reference input r(n)."
