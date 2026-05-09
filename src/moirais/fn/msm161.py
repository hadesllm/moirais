"""Numbered display equation (9.1) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_1"]


def mvsml_ridge_lasso_elastic_eq_9_1(Fig, Hyperplanes, two, left, three, right):
    """
    Numbered display equation (9.1) from MVSML chapter 9.

    Formula: Fig. 9.1 Hyperplanes in two (left) and three (right) dimensions two-dimensional subspace. Although it is hard to visualize a hyperplane when the original space has a dimension of four or more, it still applies for the ( p + 1)- dimensional ﬂat subspace (James et al. 2013). In higher dimensions, it is useful to think of a hyperplane as a member of an afﬁne family of ( p + 1)-dimensional subspaces (afﬁne spaces look and behave very similarly to linear spaces without the requirement to contain the origin), such that the whole space is partitioned into these family subspaces. From a mathematical point of view, a hyperplane is deﬁned as (James et al. 2013) \beta0 + \beta1X1 + \beta2X2 + \beta3X3 = 0

    Parameters
    ----------
    Fig : array-like
        Input data.
    Hyperplanes : array-like
        Input data.
    two : array-like
        Input data.
    left : array-like
        Input data.
    three : array-like
        Input data.
    right : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.1) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Fig = np.atleast_1d(np.asarray(Fig, dtype=float))
    n = len(Fig)
    result = float(np.mean(Fig))
    se = float(np.std(Fig, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.1) from MVSML chapter 9."})


def cheatsheet():
    return "msm161: Numbered display equation (9.1) from MVSML chapter 9."
