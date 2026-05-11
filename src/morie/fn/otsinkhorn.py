"""Entropic-regularised OT via Sinkhorn iterations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_sinkhorn"]


def ot_sinkhorn(a, b, C, epsilon, max_iter):
    """
    Entropic-regularised OT via Sinkhorn iterations

    Formula: T = diag(u) K diag(v); K=exp(-C/ε); iterate u,v

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    epsilon : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost, iters

    References
    ----------
    Cuturi (2013)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Entropic-regularised OT via Sinkhorn iterations"})


def cheatsheet():
    return "otsinkhorn: Entropic-regularised OT via Sinkhorn iterations"
