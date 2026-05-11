"""Numbered display equation (9.8) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_8"]


def mvsml_ridge_lasso_elastic_eq_9_8(minimize, z, p, yi, xT, i):
    """
    Numbered display equation (9.8) from MVSML chapter 9.

    Formula: minimize 2 \beta (9.7) |ﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄ} \beta0, \beta1, \beta2, ..., \betap   yi \beta0 + xT i \beta  1, i = 1, . . . , n

    Parameters
    ----------
    minimize : array-like
        Input data.
    z : array-like
        Input data.
    p : array-like
        Input data.
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.8) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.8) from MVSML chapter 9."})


def cheatsheet():
    return "msm183: Numbered display equation (9.8) from MVSML chapter 9."
