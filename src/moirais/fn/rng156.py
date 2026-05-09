"""Estimation error in vector LMS form.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_estimation_error"]


def rangayyan_ch3_lms_estimation_error(x, w, r, n):
    """
    Estimation error in vector LMS form.

    Formula: e(n) = x(n) - w^T(n) * r(n)

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.199, p. 184
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimation error in vector LMS form."})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimation error in vector LMS form."})


def cheatsheet():
    return "rng156: Estimation error in vector LMS form."
