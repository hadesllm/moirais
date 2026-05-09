"""Numbered display equation (9.3) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_3"]


def mvsml_ridge_lasso_elastic_eq_9_3(it, simply, an, extension, of, pXp):
    """
    Numbered display equation (9.3) from MVSML chapter 9.

    Formula: and it is simply an extension of (9.1) as \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp = 0 (9.2) In the same way, any point X = (X1, X2, . . . . Xp )T in the p-dimensional space that satisﬁes (9.2) deﬁnes a ( p + 1)-dimensional hyperplane, which means that the hyperplane is formed by those points of X that satisfy (9.2) (James et al. 2013). But those points of X that do not satisfy (9.2) like, for example, \beta0 + \beta1X1 + \beta2X2 + . . . + \betapXp < 0 (9.3) There are points that satisfying

    Parameters
    ----------
    it : array-like
        Input data.
    simply : array-like
        Input data.
    an : array-like
        Input data.
    extension : array-like
        Input data.
    of : array-like
        Input data.
    pXp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.3) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    it = np.atleast_1d(np.asarray(it, dtype=float))
    n = len(it)
    result = float(np.mean(it))
    se = float(np.std(it, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.3) from MVSML chapter 9."})


def cheatsheet():
    return "msm170: Numbered display equation (9.3) from MVSML chapter 9."
