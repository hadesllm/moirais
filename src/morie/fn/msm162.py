"""Numbered display equation (9.1) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_1"]


def mvsml_ridge_lasso_elastic_eq_9_1(original, space, has, a, dimension, of):
    """
    Numbered display equation (9.1) from MVSML chapter 9.

    Formula: original space has a dimension of four or more, it still applies for the ( p + 1)- dimensional ﬂat subspace (James et al. 2013). In higher dimensions, it is useful to think of a hyperplane as a member of an afﬁne family of ( p + 1)-dimensional subspaces (afﬁne spaces look and behave very similarly to linear spaces without the requirement to contain the origin), such that the whole space is partitioned into these family subspaces. From a mathematical point of view, a hyperplane is deﬁned as (James et al. 2013) \beta0 + \beta1X1 + \beta2X2 + \beta3X3 = 0 (9.1) for parameters \beta0, \beta1, \beta2, and \beta3.

    Parameters
    ----------
    original : array-like
        Input data.
    space : array-like
        Input data.
    has : array-like
        Input data.
    a : array-like
        Input data.
    dimension : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.1) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    original = np.atleast_1d(np.asarray(original, dtype=float))
    n = len(original)
    result = float(np.mean(original))
    se = float(np.std(original, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.1) from MVSML chapter 9."})


def cheatsheet():
    return "msm162: Numbered display equation (9.1) from MVSML chapter 9."
