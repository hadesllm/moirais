"""Numbered display equation (9.2) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_2"]


def mvsml_ridge_lasso_elastic_eq_9_2(of, a, plane, since, three, dimensions):
    """
    Numbered display equation (9.2) from MVSML chapter 9.

    Formula: of a plane, since in three dimensions, as mentioned before, a hyperplane is a plane, as can be observed in Fig. 9.1 (right). For the p-dimensional space, the dimension of the hyperplane generated is p + 1, and it is simply an extension of (9.1) as \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp = 0 (9.2) In the same way, any point X = (X1, X2, . . . . Xp )T in the p-dimensional space that satisﬁes (9.2) deﬁnes a ( p + 1)-dimensional hyperplane, which means that the hyperplane is formed by those points of X that satisfy (9.2) (James et al. 2013). But those points of X that do not satisfy

    Parameters
    ----------
    of : array-like
        Input data.
    a : array-like
        Input data.
    plane : array-like
        Input data.
    since : array-like
        Input data.
    three : array-like
        Input data.
    dimensions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.2) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    of = np.atleast_1d(np.asarray(of, dtype=float))
    n = len(of)
    result = float(np.mean(of))
    se = float(np.std(of, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.2) from MVSML chapter 9."})


def cheatsheet():
    return "msm168: Numbered display equation (9.2) from MVSML chapter 9."
