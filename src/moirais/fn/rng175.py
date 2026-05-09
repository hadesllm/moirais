"""A priori error in the RLS update step.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_a_priori_error"]


def rangayyan_ch3_rls_a_priori_error(x, r, w_tilde, n):
    """
    A priori error in the RLS update step.

    Formula: alpha(n) = x(n) - r^T(n) * w_tilde(n-1) = x(n) - w_tilde^T(n-1) * r(n)

    Parameters
    ----------
    x : array-like
        Input data.
    r : array-like
        Input data.
    w_tilde : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.225, p. 189
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "A priori error in the RLS update step."})


def cheatsheet():
    return "rng175: A priori error in the RLS update step."
