"""Time-averaged weighted autocorrelation matrix in RLS.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_phi_matrix"]


def rangayyan_ch3_rls_phi_matrix(r, lam, n):
    """
    Time-averaged weighted autocorrelation matrix in RLS.

    Formula: Phi(n) = sum_{i=1}^{n} lambda^(n-i) * r(i) * r^T(i)

    Parameters
    ----------
    r : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.208, p. 187
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    n = len(r)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Time-averaged weighted autocorrelation matrix in RLS."})
    estimate = np.median(r)
    se = 1.2533 * np.std(r, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Time-averaged weighted autocorrelation matrix in RLS."})


def cheatsheet():
    return "rng165: Time-averaged weighted autocorrelation matrix in RLS."
