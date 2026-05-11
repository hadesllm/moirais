"""Widrow-Hoff LMS tap-weight update rule.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_widrow_hoff_lms"]


def rangayyan_ch3_widrow_hoff_lms(w, mu, e, r, n):
    """
    Widrow-Hoff LMS tap-weight update rule.

    Formula: w(n+1) = w(n) + 2*mu*e(n)*r(n)

    Parameters
    ----------
    w : array-like
        Input data.
    mu : array-like
        Input data.
    e : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.203, p. 185
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Widrow-Hoff LMS tap-weight update rule."})


def cheatsheet():
    return "rng160: Widrow-Hoff LMS tap-weight update rule."
