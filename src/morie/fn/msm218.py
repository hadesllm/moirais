r"""Numbered display equation (9.34) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_34"]


def mvsml_ridge_lasso_elastic_eq_9_34(choosing, the, right, hyperplane, we, need):
    r"""
    Numbered display equation (9.34) from MVSML chapter 9.

    Formula: for choosing the right hyperplane, we need to ﬁnd: (a) a balance between the limit of the total amount of slack due to outliers, measured as Pn i=1\zetai and (b) a hyperplane with a large margin, but if the margin is larger, more outliers are possible, which implies a larger amount of slack. The optimization problem now consists of ﬁnding a hyperplane that is able to classify most of the training observations in the two classes; this can be accomplished by obtaining the solution to the following optimi- zation problem: maximize M

    Parameters
    ----------
    choosing : array-like
        Input data.
    the : array-like
        Input data.
    right : array-like
        Input data.
    hyperplane : array-like
        Input data.
    we : array-like
        Input data.
    need : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.34) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    r"""
    choosing = np.atleast_1d(np.asarray(choosing, dtype=float))
    n = len(choosing)
    result = float(np.mean(choosing))
    se = float(np.std(choosing, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.34) from MVSML chapter 9."})


def cheatsheet():
    return "msm218: Numbered display equation (9.34) from MVSML chapter 9."
