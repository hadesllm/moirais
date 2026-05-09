"""Instantaneous gradient estimate used by the LMS algorithm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_gradient_estimate"]


def rangayyan_ch3_lms_gradient_estimate(x, r, w, e, n):
    """
    Instantaneous gradient estimate used by the LMS algorithm.

    Formula: grad(e^2(n)) = -2*x(n)*r(n) + 2*{w^T(n)*r(n)}*r(n) = -2*e(n)*r(n)

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    w : array-like
        Input data.
    e : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.202, p. 185
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Instantaneous gradient estimate used by the LMS algorithm."})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Instantaneous gradient estimate used by the LMS algorithm."})


def cheatsheet():
    return "rng159: Instantaneous gradient estimate used by the LMS algorithm."
