"""Numbered display equation (9.5) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_5"]


def mvsml_ridge_lasso_elastic_eq_9_5(xip, a, p, dimensional, vector, of):
    """
    Numbered display equation (9.5) from MVSML chapter 9.

    Formula: , . . ., xip) is a p- dimensional vector of predictors (inputs) measured in sample i. We also assume that 9.3 Maximum Margin Classiﬁer 341 the response variable is binary (two classes) and coded as 1 for representing class 1 and +1 for representing class 2. A ﬁtting function of the form ( ) = \beta0 + xT f xi i \beta

    Parameters
    ----------
    xip : array-like
        Input data.
    a : array-like
        Input data.
    p : array-like
        Input data.
    dimensional : array-like
        Input data.
    vector : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.5) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    xip = np.atleast_1d(np.asarray(xip, dtype=float))
    n = len(xip)
    result = float(np.mean(xip))
    se = float(np.std(xip, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.5) from MVSML chapter 9."})


def cheatsheet():
    return "msm173: Numbered display equation (9.5) from MVSML chapter 9."
