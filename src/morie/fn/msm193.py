"""Numbered display equation (9.21) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_21"]


def mvsml_ridge_lasso_elastic_eq_9_21(z, subject, to, With, this, last):
    """
    Numbered display equation (9.21) from MVSML chapter 9.

    Formula: (9.19) |ﬄﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄﬄ} \alpha subject to \alpha  0 (9.20) With this last version of the Wolfe dual, we obtained the solution to the original optimization problem with the solution for x = 1 and \alpha = 1. Illustrative Example 9.2 x2 + y2 minimize

    Parameters
    ----------
    z : array-like
        Input data.
    subject : array-like
        Input data.
    to : array-like
        Input data.
    With : array-like
        Input data.
    this : array-like
        Input data.
    last : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.21) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.21) from MVSML chapter 9."})


def cheatsheet():
    return "msm193: Numbered display equation (9.21) from MVSML chapter 9."
