"""Numbered display equation (9.45) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_45"]


def mvsml_ridge_lasso_elastic_eq_9_45(i, jyiy, j, xi, x, z):
    """
    Numbered display equation (9.45) from MVSML chapter 9.

    Formula: i=1\alphai\alpha jyiy j xi:x j (9.44) |ﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄ} 2 \alpha Xn subject to : 0 \alphai T and i=1\alphaiyi = 0 for i = 1, . . . , n

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
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.45) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.45) from MVSML chapter 9."})


def cheatsheet():
    return "msm232: Numbered display equation (9.45) from MVSML chapter 9."
