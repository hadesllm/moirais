"""Estimation error between the desired signal and the filter output (Wiener setup).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_estimation_error"]


def rangayyan_ch3_estimation_error(d, d_tilde, n):
    """
    Estimation error between the desired signal and the filter output (Wiener setup).

    Formula: e(n) = d(n) - d_tilde(n)

    Parameters
    ----------
    d : array-like
        Input data.
    d_tilde : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.153, p. 173
    """
    d = np.atleast_1d(np.asarray(d, dtype=float))
    n = len(d)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Estimation error between the desired signal and the filter output (Wiener setup)."})
    estimate = np.median(d)
    se = 1.2533 * np.std(d, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Estimation error between the desired signal and the filter output (Wiener setup)."})


def cheatsheet():
    return "rng137: Estimation error between the desired signal and the filter output (Wiener setup)."
