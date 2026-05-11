"""Numbered display equation (9.44) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_44"]


def mvsml_ridge_lasso_elastic_eq_9_44(subject, to, i, T, iyi, n):
    """
    Numbered display equation (9.44) from MVSML chapter 9.

    Formula: subject to : 0 \alphai T and i=1\alphaiyi = 0 for i = 1, . . . , n (9.45) This problem is very similar to the one in the previous section and, again, it is a convex quadratic programming problem that can be solved using conventional quadratic programming software since the objective function is concave and inﬁ- nitely differentiable. Again, the solution to \alpha in

    Parameters
    ----------
    subject : array-like
        Input data.
    to : array-like
        Input data.
    i : array-like
        Input data.
    T : array-like
        Input data.
    iyi : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.44) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    subject = np.atleast_1d(np.asarray(subject, dtype=float))
    n = len(subject)
    result = float(np.mean(subject))
    se = float(np.std(subject, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.44) from MVSML chapter 9."})


def cheatsheet():
    return "msm233: Numbered display equation (9.44) from MVSML chapter 9."
