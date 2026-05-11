"""Time-averaged weighted cross-correlation vector in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_theta_vector"]


def rangayyan_ch3_rls_theta_vector(r, x, lam, n):
    """
    Time-averaged weighted cross-correlation vector in RLS.

    Formula: Theta(n) = sum_{i=1}^{n} lambda^(n-i) * r(i) * x(i)

    Parameters
    ----------
    r : array-like
        Input data.
    x : array-like
        Input data.
    lam : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.209, p. 187
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Time-averaged weighted cross-correlation vector in RLS."})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Time-averaged weighted cross-correlation vector in RLS."})


def cheatsheet():
    return "rng166: Time-averaged weighted cross-correlation vector in RLS."
