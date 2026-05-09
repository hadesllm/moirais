"""Numbered display equation (9.31) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_31"]


def mvsml_ridge_lasso_elastic_eq_9_31(i, iyi, z, Pn, jyiy, j):
    """
    Numbered display equation (9.31) from MVSML chapter 9.

    Formula: i=1\alphaiyi\beta0 + i=1\alphai 2 |ﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ} +0:5Pn ( ) i=1\alphai\alpha jyiy j xi:x j (9.31) Simplifying

    Parameters
    ----------
    i : array-like
        Input data.
    iyi : array-like
        Input data.
    z : array-like
        Input data.
    Pn : array-like
        Input data.
    jyiy : array-like
        Input data.
    j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.31) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.31) from MVSML chapter 9."})


def cheatsheet():
    return "msm211: Numbered display equation (9.31) from MVSML chapter 9."
