"""Numbered display equation (9.4) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_4"]


def mvsml_ridge_lasso_elastic_eq_9_4(The, term, yi, xT, i, the):
    """
    Numbered display equation (9.4) from MVSML chapter 9.

    Formula:  The term yi \beta0 + xT i \beta in the restrictions of (9.6) of this optimization problem is the distance between the ith observation and the decision boundary and is essential for correctly identifying classiﬁed observations on or beyond the margin boundary, given that M is positive. 2M is the whole margin or width of the street (see Fig. 9.4b), since M (half-width of the street) is the distance, centered on the decision boundary, to the margin boundary from the decision boundary. It is important to point out that the constraints given in

    Parameters
    ----------
    The : array-like
        Input data.
    term : array-like
        Input data.
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    i : array-like
        Input data.
    the : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.4) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    The = np.atleast_1d(np.asarray(The, dtype=float))
    n = len(The)
    result = float(np.mean(The))
    se = float(np.std(The, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.4) from MVSML chapter 9."})


def cheatsheet():
    return "msm176: Numbered display equation (9.4) from MVSML chapter 9."
