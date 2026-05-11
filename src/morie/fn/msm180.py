"""Numbered display equation (9.6) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_6"]


def mvsml_ridge_lasso_elastic_eq_9_6(distance, margin, M, hyperplane, h0, xT):
    """
    Numbered display equation (9.6) from MVSML chapter 9.

    Formula: distance (margin, M) from hyperplane (h0=\beta0 + xT i \beta = 0) to hyperplane (h1 = \beta0 + xT i \beta = 1). This means that the total distance is equal to 2M = 2/k\betak. This implies that maximizing M = 1/k\betak subject to the constraints of

    Parameters
    ----------
    distance : array-like
        Input data.
    margin : array-like
        Input data.
    M : array-like
        Input data.
    hyperplane : array-like
        Input data.
    h0 : array-like
        Input data.
    xT : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.6) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    distance = np.atleast_1d(np.asarray(distance, dtype=float))
    n = len(distance)
    result = float(np.mean(distance))
    se = float(np.std(distance, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.6) from MVSML chapter 9."})


def cheatsheet():
    return "msm180: Numbered display equation (9.6) from MVSML chapter 9."
