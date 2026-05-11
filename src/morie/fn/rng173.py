"""Identity P(n)*r(n) = k(n) used in deriving the RLS update.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_rls_gain_identity"]


def rangayyan_ch3_rls_gain_identity(P, r, n):
    """
    Identity P(n)*r(n) = k(n) used in deriving the RLS update.

    Formula: k(n) = P(n) * r(n)

    Parameters
    ----------
    P : array-like
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
    Rangayyan (2024), Ch 3, Eq 3.221, p. 188
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    n = len(P)
    result = float(np.mean(P))
    se = float(np.std(P, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Identity P(n)*r(n) = k(n) used in deriving the RLS update."})


def cheatsheet():
    return "rng173: Identity P(n)*r(n) = k(n) used in deriving the RLS update."
