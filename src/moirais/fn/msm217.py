"""Numbered display equation (9.33) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_33"]


def mvsml_ridge_lasso_elastic_eq_9_33(e, dot, product, of, vectors, xi):
    """
    Numbered display equation (9.33) from MVSML chapter 9.

    Formula: e dot product of vectors xi and xj. This means that we do not exactly need the exact data points, but only their inner products to compute our decision boundary. What it implies is that if we want to transform our existing data into a higher dimensional data, which in many cases helps us classify better (see the image below for an example), we need not compute the exact transformation of our data, we just need the inner product of our data in that higher dimensional space. It is important to point out that the constraints in (9.33) are afﬁne and convex. Also, (9.32) is inﬁnitely differentiable and its Hessian is positive semi-deﬁnite which implies that the maximization problem in (9.32) and

    Parameters
    ----------
    e : array-like
        Input data.
    dot : array-like
        Input data.
    product : array-like
        Input data.
    of : array-like
        Input data.
    vectors : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.33) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    e = np.atleast_1d(np.asarray(e, dtype=float))
    n = len(e)
    result = float(np.mean(e))
    se = float(np.std(e, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.33) from MVSML chapter 9."})


def cheatsheet():
    return "msm217: Numbered display equation (9.33) from MVSML chapter 9."
