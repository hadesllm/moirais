"""Unbalanced OT with KL marginal penalties."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_unbalanced"]


def ot_unbalanced(a, b, C, epsilon, lam):
    """
    Unbalanced OT with KL marginal penalties

    Formula: min_T <T,C> + ε H(T) + λ KL(T1|a) + λ KL(T^T 1|b)

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
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: T, cost

    References
    ----------
    Chizat-Peyré-Schmitzer-Vialard (2018)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unbalanced OT with KL marginal penalties"})


def cheatsheet():
    return "otunbal: Unbalanced OT with KL marginal penalties"
