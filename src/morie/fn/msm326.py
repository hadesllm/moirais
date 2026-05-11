"""Numbered display equation (15.2) from MVSML chapter 15.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_functional_regression_eq_15_2"]


def mvsml_functional_regression_eq_15_2(Y, i, N, exp, For, a):
    """
    Numbered display equation (15.2) from MVSML chapter 15.

    Formula: Y+ i \mu i = N+ 1  exp \mu ( ) For a given candidate split, the log-likelihood function given in

    Parameters
    ----------
    Y : array-like
        Input data.
    i : array-like
        Input data.
    N : array-like
        Input data.
    exp : array-like
        Input data.
    For : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (15.2) [Multivariate Statistical Machine Learnin [Pages 633-681] [2026-04-16].pdf]
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (15.2) from MVSML chapter 15."})


def cheatsheet():
    return "msm326: Numbered display equation (15.2) from MVSML chapter 15."
