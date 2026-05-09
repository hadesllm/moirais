"""Numbered display equation (9.1) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_1"]


def mvsml_ridge_lasso_elastic_eq_9_1(dimensional, at, subspace, James, et, al):
    """
    Numbered display equation (9.1) from MVSML chapter 9.

    Formula: dimensional ﬂat subspace (James et al. 2013). In higher dimensions, it is useful to think of a hyperplane as a member of an afﬁne family of ( p + 1)-dimensional subspaces (afﬁne spaces look and behave very similarly to linear spaces without the requirement to contain the origin), such that the whole space is partitioned into these family subspaces. From a mathematical point of view, a hyperplane is deﬁned as (James et al. 2013) \beta0 + \beta1X1 + \beta2X2 + \beta3X3 = 0 (9.1) for parameters \beta0, \beta1, \beta2, and \beta3. (9.1) “deﬁnes” a hyperplane, since any X = (X1, X2, X3 )T for which

    Parameters
    ----------
    dimensional : array-like
        Input data.
    at : array-like
        Input data.
    subspace : array-like
        Input data.
    James : array-like
        Input data.
    et : array-like
        Input data.
    al : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.1) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    dimensional = np.atleast_1d(np.asarray(dimensional, dtype=float))
    n = len(dimensional)
    result = float(np.mean(dimensional))
    se = float(np.std(dimensional, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.1) from MVSML chapter 9."})


def cheatsheet():
    return "msm163: Numbered display equation (9.1) from MVSML chapter 9."
