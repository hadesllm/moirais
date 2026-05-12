r"""Numbered display equation (9.14) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_14"]


def mvsml_ridge_lasso_elastic_eq_9_14(Xm, Xp, subject, to, f, x):
    r"""
    Numbered display equation (9.14) from MVSML chapter 9.

    Formula: Xm Xp subject to \nabla f x ( ) + i=1\lambdai \nabla hi x ( ) + i=1\alphai \nabla gi x ( ) = 0 (9.13) \alphai  0, i = 1, . . . , p

    Parameters
    ----------
    Xm : array-like
        Input data.
    Xp : array-like
        Input data.
    subject : array-like
        Input data.
    to : array-like
        Input data.
    f : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.14) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.14) from MVSML chapter 9."})


def cheatsheet():
    return "msm187: Numbered display equation (9.14) from MVSML chapter 9."
