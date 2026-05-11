"""Inverse of the ALR transform."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aitchison_alr_inverse"]


def aitchison_alr_inverse(y):
    """
    Inverse of the ALR transform

    Formula: x = C([exp(y), 1])

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Aitchison (1986) §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse of the ALR transform"})


def cheatsheet():
    return "aitalri: Inverse of the ALR transform"
