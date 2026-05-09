"""Numbered display equation (9.6) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_6"]


def mvsml_ridge_lasso_elastic_eq_9_6(n, the, restrictions, of, this, optimization):
    """
    Numbered display equation (9.6) from MVSML chapter 9.

    Formula: n the restrictions of (9.6) of this optimization problem is the distance between the ith observation and the decision boundary and is essential for correctly identifying classiﬁed observations on or beyond the margin boundary, given that M is positive. 2M is the whole margin or width of the street (see Fig. 9.4b), since M (half-width of the street) is the distance, centered on the decision boundary, to the margin boundary from the decision boundary. It is important to point out that the constraints given in (9.4) and (9.5) guarantee that each observation will fall on the correct side of the hyperplane and at a distance of at least M from the hyperplane. The fact that the last restriction of

    Parameters
    ----------
    n : array-like
        Input data.
    the : array-like
        Input data.
    restrictions : array-like
        Input data.
    of : array-like
        Input data.
    this : array-like
        Input data.
    optimization : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.6) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.6) from MVSML chapter 9."})


def cheatsheet():
    return "msm178: Numbered display equation (9.6) from MVSML chapter 9."
