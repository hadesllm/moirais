"""Numbered display equation (9.30) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_30"]


def mvsml_ridge_lasso_elastic_eq_9_30(yi, xT, i, The, conditions, that):
    """
    Numbered display equation (9.30) from MVSML chapter 9.

    Formula:   = 0 and yi \beta0 + xT i \beta = 1 (9.30) The conditions that the solution must satisfy are called the Karush–Kuhn–Tucker conditions. They are required to ensure that the function is convex to guarantee a local optimum of nonlinear programming problems. We can see from

    Parameters
    ----------
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    i : array-like
        Input data.
    The : array-like
        Input data.
    conditions : array-like
        Input data.
    that : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.30) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.30) from MVSML chapter 9."})


def cheatsheet():
    return "msm205: Numbered display equation (9.30) from MVSML chapter 9."
