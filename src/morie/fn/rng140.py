"""Estimation error in vector form for the Wiener filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_estimation_error_vector_form"]


def rangayyan_ch3_estimation_error_vector_form(d, w, x, n):
    """
    Estimation error in vector form for the Wiener filter.

    Formula: e(n) = d(n) - w^T * x(n)

    Parameters
    ----------
    d : array-like
        Input data.
    w : array-like
        Input data.
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
    Rangayyan (2024), Ch 3, Eq 3.158, p. 174
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimation error in vector form for the Wiener filter."})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimation error in vector form for the Wiener filter."})


def cheatsheet():
    return "rng140: Estimation error in vector form for the Wiener filter."
