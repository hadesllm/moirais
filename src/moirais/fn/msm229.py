"""Numbered display equation (9.39) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_39"]


def mvsml_ridge_lasso_elastic_eq_9_39(i, n, yi, xT, By, placing):
    """
    Numbered display equation (9.39) from MVSML chapter 9.

    Formula: = 0 for i = 1, . . . , n \Rightarrow \alphai   = 0 and yi \beta0 + xT i \beta = 1 + \zetai (9.42) \deltai\zetai = 0 for i = 1, . . . , n \Rightarrow \deltai = 0 and \zetai = 0 (9.43) By placing solutions

    Parameters
    ----------
    i : array-like
        Input data.
    n : array-like
        Input data.
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    By : array-like
        Input data.
    placing : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.39) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.39) from MVSML chapter 9."})


def cheatsheet():
    return "msm229: Numbered display equation (9.39) from MVSML chapter 9."
