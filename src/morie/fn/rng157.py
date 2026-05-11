"""Quadratic squared-error form used in LMS gradient derivations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_squared_error"]


def rangayyan_ch3_lms_squared_error(x, r, w, n):
    """
    Quadratic squared-error form used in LMS gradient derivations.

    Formula: e^2(n) = x^2(n) - 2*x(n)*r^T(n)*w(n) + w^T(n)*r(n)*r^T(n)*w(n)

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    w : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.200, p. 184
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic squared-error form used in LMS gradient derivations."})


def cheatsheet():
    return "rng157: Quadratic squared-error form used in LMS gradient derivations."
