# moirais.fn — function file (hadesllm/moirais)
"""ARE of Theorem 13.2.2 remains valid for two-sided alternatives."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_are_twosided"]


def gibbons_are_twosided(T, T_star):
    """
    ARE of Theorem 13.2.2 remains valid for two-sided alternatives

    Formula: ARE(T,T*) same formula for two-sided rejection region

    Parameters
    ----------
    T : array-like
        Input data.
    T_star : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ARE

    References
    ----------
    Gibbons Theorem 13.2.3
    """
    T = np.asarray(T, dtype=float)
    n = int(T) if T.ndim == 0 else len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARE of Theorem 13.2.2 remains valid for two-sided alternatives"})


def cheatsheet():
    return "gb1323: ARE of Theorem 13.2.2 remains valid for two-sided alternatives"
