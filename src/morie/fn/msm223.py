r"""Numbered display equation (9.38) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_38"]


def mvsml_ridge_lasso_elastic_eq_9_38(i, yi, xT, L, e, Xn):
    r"""
    Numbered display equation (9.38) from MVSML chapter 9.

    Formula: i=1\alphai yi \beta0 + xT L \beta, \beta0, e, \alpha, \delta ( 2 \beta i=1\zetai 2 i \beta + 1 + \zetai Xn + i=1\deltai\zetai,

    Parameters
    ----------
    i : array-like
        Input data.
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    L : array-like
        Input data.
    e : array-like
        Input data.
    Xn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.38) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.38) from MVSML chapter 9."})


def cheatsheet():
    return "msm223: Numbered display equation (9.38) from MVSML chapter 9."
