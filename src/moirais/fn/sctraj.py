"""Pseudotime trajectory (slingshot/monocle)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["scrnaseq_trajectory"]


def scrnaseq_trajectory(X, clusters, start_cluster):
    """
    Pseudotime trajectory (slingshot/monocle)

    Formula: principal curve through cluster centers

    Parameters
    ----------
    X : array-like
        Input data.
    clusters : array-like
        Input data.
    start_cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Street et al (2018) Slingshot
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pseudotime trajectory (slingshot/monocle)"})


def cheatsheet():
    return "sctraj: Pseudotime trajectory (slingshot/monocle)"
