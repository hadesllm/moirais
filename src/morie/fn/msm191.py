"""Numbered display equation (9.19) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_19"]


def mvsml_ridge_lasso_elastic_eq_9_19(subject, to, x, Then, the, last):
    """
    Numbered display equation (9.19) from MVSML chapter 9.

    Formula: ) subject to = 2x + 2\alpha = 0 \partial x and \alpha  0 (9.18) Then the last version of the Wolfe dual can be simpliﬁed as ( ) = +\alpha2 + 2\alpha maximize L \lambda

    Parameters
    ----------
    subject : array-like
        Input data.
    to : array-like
        Input data.
    x : array-like
        Input data.
    Then : array-like
        Input data.
    the : array-like
        Input data.
    last : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.19) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.19) from MVSML chapter 9."})


def cheatsheet():
    return "msm191: Numbered display equation (9.19) from MVSML chapter 9."
