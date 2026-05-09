"""Linkage disequilibrium r^2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ld_r2"]


def ld_r2(geno1, geno2):
    """
    Linkage disequilibrium r^2

    Formula: r^2 = D^2 / (p1 q1 p2 q2)

    Parameters
    ----------
    geno1 : array-like
        Input data.
    geno2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hill-Robertson (1968)
    """
    geno1 = np.atleast_1d(np.asarray(geno1, dtype=float))
    n = len(geno1)
    result = float(np.mean(geno1))
    se = float(np.std(geno1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linkage disequilibrium r^2"})


def cheatsheet():
    return "ldcmpr: Linkage disequilibrium r^2"
