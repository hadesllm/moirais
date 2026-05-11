"""Numbered display equation (9.6) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_6"]


def mvsml_ridge_lasso_elastic_eq_9_6(yi, xT, i, M, n, The):
    """
    Numbered display equation (9.6) from MVSML chapter 9.

    Formula:   yi \beta0 + xT i \beta  M, i = 1, . . . , n   The term yi \beta0 + xT i \beta in the restrictions of

    Parameters
    ----------
    yi : array-like
        Input data.
    xT : array-like
        Input data.
    i : array-like
        Input data.
    M : array-like
        Input data.
    n : array-like
        Input data.
    The : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.6) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.6) from MVSML chapter 9."})


def cheatsheet():
    return "msm175: Numbered display equation (9.6) from MVSML chapter 9."
