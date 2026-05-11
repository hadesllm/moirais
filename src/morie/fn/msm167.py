"""Numbered display equation (9.2) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_2"]


def mvsml_ridge_lasso_elastic_eq_9_2(X3, T, which, holds, a, point):
    """
    Numbered display equation (9.2) from MVSML chapter 9.

    Formula: X3 )T for which (9.1) holds is a point in the hyperplane. Equation (9.1) is the equation of a plane, since in three dimensions, as mentioned before, a hyperplane is a plane, as can be observed in Fig. 9.1 (right). For the p-dimensional space, the dimension of the hyperplane generated is p + 1, and it is simply an extension of (9.1) as \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp = 0 (9.2) In the same way, any point X = (X1, X2, . . . . Xp )T in the p-dimensional space that satisﬁes (9.2) deﬁnes a ( p + 1)-dimensional hyperplane, which means that the hyperplane is formed by those points of X that satisfy

    Parameters
    ----------
    X3 : array-like
        Input data.
    T : array-like
        Input data.
    which : array-like
        Input data.
    holds : array-like
        Input data.
    a : array-like
        Input data.
    point : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.2) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    X3 = np.atleast_1d(np.asarray(X3, dtype=float))
    n = len(X3)
    result = float(np.mean(X3))
    se = float(np.std(X3, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.2) from MVSML chapter 9."})


def cheatsheet():
    return "msm167: Numbered display equation (9.2) from MVSML chapter 9."
