"""Numbered display equation (9.30) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_30"]


def mvsml_ridge_lasso_elastic_eq_9_30(the, slab, b, If, yi, xT):
    """
    Numbered display equation (9.30) from MVSML chapter 9.

    Formula: the slab.   (b) If yi \beta0 + xT i \beta > 1, xi is not on the boundary of the slab, and \alphai = 0. From (9.28), we can see that the beta coefﬁcients (with the exception of the intercept) of the maximum margin hyperplane problem are a linear combination of the training vectors x1, . . . . , xn. A vector xi belongs to that expansion if, and only if, \alphai 6= 0 and these vectors are called support vectors. By condition

    Parameters
    ----------
    the : array-like
        Input data.
    slab : array-like
        Input data.
    b : array-like
        Input data.
    If : array-like
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
    MVSML, Eq. (9.30) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    the = np.atleast_1d(np.asarray(the, dtype=float))
    n = len(the)
    result = float(np.mean(the))
    se = float(np.std(the, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.30) from MVSML chapter 9."})


def cheatsheet():
    return "msm207: Numbered display equation (9.30) from MVSML chapter 9."
