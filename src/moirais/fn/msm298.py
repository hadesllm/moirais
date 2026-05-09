"""Numbered display equation (14.5) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_5"]


def mvsml_convolutional_nn_eq_14_5(where, a, vector, of, dimension, n):
    """
    Numbered display equation (14.5) from MVSML chapter 14.

    Formula: (14.14) where 1n is a vector of dimension n  1 with all its entries equal to T and xi = xi1, . . . , xiL1 T, i = 1, . . . , n = n1 + ⋯+ nI, 1, X = x1, . . . , xn = = as deﬁned in (14.4) and

    Parameters
    ----------
    where : array-like
        Input data.
    a : array-like
        Input data.
    vector : array-like
        Input data.
    of : array-like
        Input data.
    dimension : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.5) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.5) from MVSML chapter 14."})


def cheatsheet():
    return "msm298: Numbered display equation (14.5) from MVSML chapter 14."
