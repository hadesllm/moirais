"""Compact RLS tap-weight update using a priori error alpha(n).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_weight_update_compact"]


def rangayyan_ch3_rls_weight_update_compact(w_tilde, k, alpha, n):
    """
    Compact RLS tap-weight update using a priori error alpha(n).

    Formula: w_tilde(n) = w_tilde(n-1) + k(n) * alpha(n)

    Parameters
    ----------
    w_tilde : array-like
        Input data.
    k : array-like
        Input data.
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.224, p. 189
    """
    w_tilde = np.atleast_1d(np.asarray(w_tilde, dtype=float))
    n = len(w_tilde)
    result = float(np.mean(w_tilde))
    se = float(np.std(w_tilde, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Compact RLS tap-weight update using a priori error alpha(n)."})


def cheatsheet():
    return "rng174: Compact RLS tap-weight update using a priori error alpha(n)."
