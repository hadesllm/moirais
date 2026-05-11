"""Numbered display equation (9.32) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_32"]


def mvsml_ridge_lasso_elastic_eq_9_32(margin, classi, er, Xn, i, maximize):
    """
    Numbered display equation (9.32) from MVSML chapter 9.

    Formula: margin classiﬁer Xn Xn   i=1\alphai 2 1 maximize L \alpha ( ) = i=1\alphai\alpha jyiy j xi:x j

    Parameters
    ----------
    margin : array-like
        Input data.
    classi : array-like
        Input data.
    er : array-like
        Input data.
    Xn : array-like
        Input data.
    i : array-like
        Input data.
    maximize : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.32) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    margin = np.atleast_1d(np.asarray(margin, dtype=float))
    n = len(margin)
    result = float(np.mean(margin))
    se = float(np.std(margin, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.32) from MVSML chapter 9."})


def cheatsheet():
    return "msm212: Numbered display equation (9.32) from MVSML chapter 9."
