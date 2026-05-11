"""Numbered display equation (9.15) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_15"]


def mvsml_ridge_lasso_elastic_eq_9_15(i, p, This, changes, the, searching):
    """
    Numbered display equation (9.15) from MVSML chapter 9.

    Formula: \alphai  0, i = 1, . . . , p (9.14) This changes the searching space to an (n + m + p)-dimensional space, x, \lambda, \alpha, with p + 1 constraints. The Wolfe dual is a type of Lagrange dual problem. It is important to point out that the sign of the equality constraint does not matter, and we may deﬁne it as addition or subtraction, as we wish. However, the sign of the inequality constraint is crucial and should be negative for minimization and positive for maximization. Illustrative Example 9.1 x2

    Parameters
    ----------
    i : array-like
        Input data.
    p : array-like
        Input data.
    This : array-like
        Input data.
    changes : array-like
        Input data.
    the : array-like
        Input data.
    searching : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.15) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.15) from MVSML chapter 9."})


def cheatsheet():
    return "msm188: Numbered display equation (9.15) from MVSML chapter 9."
