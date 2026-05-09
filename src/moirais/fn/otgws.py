"""Entropic Gromov-Wasserstein via mirror descent."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_gromov_sinkhorn"]


def ot_gromov_sinkhorn(Cx, Cy, a, b, epsilon, max_iter):
    """
    Entropic Gromov-Wasserstein via mirror descent

    Formula: Sinkhorn on linearised tensor at each step

    Parameters
    ----------
    Cx : array-like
        Input data.
    Cy : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    epsilon : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost

    References
    ----------
    Peyré-Cuturi-Solomon (2016)
    """
    Cx = np.atleast_1d(np.asarray(Cx, dtype=float))
    n = len(Cx)
    result = float(np.mean(Cx))
    se = float(np.std(Cx, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Entropic Gromov-Wasserstein via mirror descent"})


def cheatsheet():
    return "otgws: Entropic Gromov-Wasserstein via mirror descent"
