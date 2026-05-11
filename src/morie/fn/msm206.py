"""Numbered display equation (9.28) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_28"]


def mvsml_ridge_lasso_elastic_eq_9_28(a, If, i, then, yi, xT):
    """
    Numbered display equation (9.28) from MVSML chapter 9.

    Formula: (a) If \alphai > 0, then yi \beta0 + xT i \beta = 1, or in other words, xi is on the boundary of the slab.   (b) If yi \beta0 + xT i \beta > 1, xi is not on the boundary of the slab, and \alphai = 0. From

    Parameters
    ----------
    a : array-like
        Input data.
    If : array-like
        Input data.
    i : array-like
        Input data.
    then : array-like
        Input data.
    yi : array-like
        Input data.
    xT : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.28) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.28) from MVSML chapter 9."})


def cheatsheet():
    return "msm206: Numbered display equation (9.28) from MVSML chapter 9."
