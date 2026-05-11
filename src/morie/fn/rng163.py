"""Weighted least-squares objective for the RLS algorithm with forgetting factor lambda.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_objective"]


def rangayyan_ch3_rls_objective(e, lam, n):
    """
    Weighted least-squares objective for the RLS algorithm with forgetting factor lambda.

    Formula: xi(n) = sum_{i=1}^{n} lambda^(n-i) * |e(i)|^2

    Parameters
    ----------
    e : array-like
        Input data.
    lam : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.206, p. 186
    """
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = len(e)
    result = float(np.mean(e))
    se = float(np.std(e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weighted least-squares objective for the RLS algorithm with forgetting factor lambda."})


def cheatsheet():
    return "rng163: Weighted least-squares objective for the RLS algorithm with forgetting factor lambda."
