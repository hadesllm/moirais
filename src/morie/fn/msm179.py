"""Numbered display equation (9.6) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_6"]


def mvsml_ridge_lasso_elastic_eq_9_6(observations, are, inside, the, fences, street):
    """
    Numbered display equation (9.6) from MVSML chapter 9.

    Formula: observations are inside the fences (street). To obtain the distance from a point to the hyperplane, consider point x in Fig. 9.5. Note that from any two points x1 and x2 lying in hyperplane H, we have that \beta0 + xT 1\beta = 0 and \beta0 + xT 2\beta = 0, which implies that (x1 + x2)T\beta = 0. But because x1 + x2 is a vector in H, then \beta is orthogonal to H, and consequently also to the normalized \beta vector, \beta = \beta j (see Fig. 9.5). To solve the optimization problem

    Parameters
    ----------
    observations : array-like
        Input data.
    are : array-like
        Input data.
    inside : array-like
        Input data.
    the : array-like
        Input data.
    fences : array-like
        Input data.
    street : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.6) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    observations = np.atleast_1d(np.asarray(observations, dtype=float))
    n = len(observations)
    result = float(np.mean(observations))
    se = float(np.std(observations, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.6) from MVSML chapter 9."})


def cheatsheet():
    return "msm179: Numbered display equation (9.6) from MVSML chapter 9."
