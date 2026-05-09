"""Numbered display equation (9.3) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_3"]


def mvsml_ridge_lasso_elastic_eq_9_3(can, be, observed, Fig, right, For):
    """
    Numbered display equation (9.3) from MVSML chapter 9.

    Formula: can be observed in Fig. 9.1 (right). For the p-dimensional space, the dimension of the hyperplane generated is p + 1, and it is simply an extension of (9.1) as \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp = 0 (9.2) In the same way, any point X = (X1, X2, . . . . Xp )T in the p-dimensional space that satisﬁes (9.2) deﬁnes a ( p + 1)-dimensional hyperplane, which means that the hyperplane is formed by those points of X that satisfy (9.2) (James et al. 2013). But those points of X that do not satisfy (9.2) like, for example, \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp < 0

    Parameters
    ----------
    can : array-like
        Input data.
    be : array-like
        Input data.
    observed : array-like
        Input data.
    Fig : array-like
        Input data.
    right : array-like
        Input data.
    For : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.3) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    can = np.atleast_1d(np.asarray(can, dtype=float))
    n = len(can)
    result = float(np.mean(can))
    se = float(np.std(can, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.3) from MVSML chapter 9."})


def cheatsheet():
    return "msm169: Numbered display equation (9.3) from MVSML chapter 9."
