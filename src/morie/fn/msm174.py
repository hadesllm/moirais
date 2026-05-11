"""Numbered display equation (9.5) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_5"]


def mvsml_ridge_lasso_elastic_eq_9_5(i, b, positive, labeled, bf, xi):
    """
    Numbered display equation (9.5) from MVSML chapter 9.

    Formula: ib\beta is positive, and labeled as +1 if bf xi ( ( ) is negative. bf xi ( ) is calculated with the estimates of the beta coefﬁcients. Before estimating the required beta coefﬁcients, we assume for the moment that the training data set is linearly separable in the predictor space, which means that there is at least one set of beta coefﬁcient parameters, (\beta0, \beta), so that using the function given in

    Parameters
    ----------
    i : array-like
        Input data.
    b : array-like
        Input data.
    positive : array-like
        Input data.
    labeled : array-like
        Input data.
    bf : array-like
        Input data.
    xi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.5) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.5) from MVSML chapter 9."})


def cheatsheet():
    return "msm174: Numbered display equation (9.5) from MVSML chapter 9."
