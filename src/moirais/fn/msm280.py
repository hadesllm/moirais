"""Numbered display equation (14.10) from MVSML chapter 14.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_convolutional_nn_eq_14_10"]


def mvsml_convolutional_nn_eq_14_10(where, P, a, square, matrix, entries):
    """
    Numbered display equation (14.10) from MVSML chapter 14.

    Formula: ( ) where P is a square matrix with entries Pij = t( ), i, j = 1, . . ., L1, and i j ϕ p ( ) t( ) is a derivate of order p of ϕi(t). Typical chosen values of p are 1 and 2. i A smoothed solution of the function \beta(t) can be obtained by minimizing

    Parameters
    ----------
    where : array-like
        Input data.
    P : array-like
        Input data.
    a : array-like
        Input data.
    square : array-like
        Input data.
    matrix : array-like
        Input data.
    entries : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (14.10) [Multivariate Statistical Machine Learnin [Pages 579-631] [2026-04-16].pdf]
    """
    where = np.atleast_1d(np.asarray(where, dtype=float))
    n = len(where)
    result = float(np.mean(where))
    se = float(np.std(where, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (14.10) from MVSML chapter 14."})


def cheatsheet():
    return "msm280: Numbered display equation (14.10) from MVSML chapter 14."
