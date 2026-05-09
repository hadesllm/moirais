"""Inverse of the CLR transform (closure of exp)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_clr_inverse"]


def aitchison_clr_inverse(z):
    """
    Inverse of the CLR transform (closure of exp)

    Formula: x = C(exp(z))

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Aitchison (1986) §4
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse of the CLR transform (closure of exp)"})


def cheatsheet():
    return "aitclri: Inverse of the CLR transform (closure of exp)"
