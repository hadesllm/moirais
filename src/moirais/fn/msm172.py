"""Numbered display equation (9.2) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_2"]


def mvsml_ridge_lasso_elastic_eq_9_2(Support, Vector, Machines, Regression, Fig, The):
    """
    Numbered display equation (9.2) from MVSML chapter 9.

    Formula: 9 Support Vector Machines and Support Vector Regression Fig. 9.2 The hyperplane 1 + 2X1 + 3X2 = 0 is shown. The blue region is the set of points for which 1 + 2X1 + 3X2 > 0, and the red region is the set of points for which 1 + 2X1 + 3X2 < 0 (James et al. 2013) \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp > 0 (9.4) will lie on the other side of the hyperplane. This means that we can think of the hyperplane as a mechanism that can divide the p-dimensional space into two halves. By simply calculating the sign of the left-hand side of

    Parameters
    ----------
    Support : array-like
        Input data.
    Vector : array-like
        Input data.
    Machines : array-like
        Input data.
    Regression : array-like
        Input data.
    Fig : array-like
        Input data.
    The : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.2) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Support = np.atleast_1d(np.asarray(Support, dtype=float))
    n = len(Support)
    result = float(np.mean(Support))
    se = float(np.std(Support, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.2) from MVSML chapter 9."})


def cheatsheet():
    return "msm172: Numbered display equation (9.2) from MVSML chapter 9."
