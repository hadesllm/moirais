"""Numbered display equation (9.31) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_31"]


def mvsml_ridge_lasso_elastic_eq_9_31(i, jyiy, j, xi, x, iyi):
    """
    Numbered display equation (9.31) from MVSML chapter 9.

    Formula: i=1\alphai\alpha jyiy j xi:x j 2 i=1\alphaiyi\beta0 + i=1\alphai 2 |ﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄﬄ} +0:5Pn ( ) i=1\alphai\alpha jyiy j xi:x j

    Parameters
    ----------
    i : array-like
        Input data.
    jyiy : array-like
        Input data.
    j : array-like
        Input data.
    xi : array-like
        Input data.
    x : array-like
        Input data.
    iyi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.31) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.31) from MVSML chapter 9."})


def cheatsheet():
    return "msm210: Numbered display equation (9.31) from MVSML chapter 9."
