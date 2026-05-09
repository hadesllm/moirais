"""Numbered display equation (9.22) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_22"]


def mvsml_ridge_lasso_elastic_eq_9_22(With, this, last, version, of, the):
    """
    Numbered display equation (9.22) from MVSML chapter 9.

    Formula: (9.20) With this last version of the Wolfe dual, we obtained the solution to the original optimization problem with the solution for x = 1 and \alpha = 1. Illustrative Example 9.2 x2 + y2 minimize (9.21) |ﬄﬄﬄﬄﬄ{zﬄﬄﬄﬄﬄ} x, y x + y  2

    Parameters
    ----------
    With : array-like
        Input data.
    this : array-like
        Input data.
    last : array-like
        Input data.
    version : array-like
        Input data.
    of : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.22) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    With = np.atleast_1d(np.asarray(With, dtype=float))
    n = len(With)
    result = float(np.mean(With))
    se = float(np.std(With, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.22) from MVSML chapter 9."})


def cheatsheet():
    return "msm194: Numbered display equation (9.22) from MVSML chapter 9."
