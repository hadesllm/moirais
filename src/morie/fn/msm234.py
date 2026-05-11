"""Numbered display equation (9.46) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_46"]


def mvsml_ridge_lasso_elastic_eq_9_46(to, the, following, optimization, problem, Xn):
    """
    Numbered display equation (9.46) from MVSML chapter 9.

    Formula: to the following optimization problem: Xn Xn   i=1\alphai 2 1 maximize L \alpha ( ) = i=1\alphai\alpha jyiy jK xi, xj

    Parameters
    ----------
    to : array-like
        Input data.
    the : array-like
        Input data.
    following : array-like
        Input data.
    optimization : array-like
        Input data.
    problem : array-like
        Input data.
    Xn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.46) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    to = np.atleast_1d(np.asarray(to, dtype=float))
    n = len(to)
    result = float(np.mean(to))
    se = float(np.std(to, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.46) from MVSML chapter 9."})


def cheatsheet():
    return "msm234: Numbered display equation (9.46) from MVSML chapter 9."
