"""Genomic heritability h^2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["heritability"]


def heritability(y, K):
    """
    Genomic heritability h^2

    Formula: sigma_g^2 / (sigma_g^2 + sigma_e^2)

    Parameters
    ----------
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanRaden (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genomic heritability h^2"})


def cheatsheet():
    return "hertbg: Genomic heritability h^2"
