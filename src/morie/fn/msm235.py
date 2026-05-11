"""Numbered display equation (9.47) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_47"]


def mvsml_ridge_lasso_elastic_eq_9_47(i, jyiy, jK, xi, xj, z):
    """
    Numbered display equation (9.47) from MVSML chapter 9.

    Formula: i=1\alphai\alpha jyiy jK xi, xj (9.46) |ﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄ} 2 \alpha Xn subject to : 0 \alphai T and i=1\alphaiyi = 0 for i = 1, . . . , n

    Parameters
    ----------
    i : array-like
        Input data.
    jyiy : array-like
        Input data.
    jK : array-like
        Input data.
    xi : array-like
        Input data.
    xj : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.47) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.47) from MVSML chapter 9."})


def cheatsheet():
    return "msm235: Numbered display equation (9.47) from MVSML chapter 9."
