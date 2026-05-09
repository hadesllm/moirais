"""Numbered display equation (9.9) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_9"]


def mvsml_ridge_lasso_elastic_eq_9_9(T, dual, result, which, explained, below):
    """
    Numbered display equation (9.9) from MVSML chapter 9.

    Formula: 2 \betaT \beta. dual result, which is explained below. Also, remember that 1 2 \beta 9.3.2 Wolfe Dual Assume we have the following general optimization problem: x 2 Rn minimize f x ( )

    Parameters
    ----------
    T : array-like
        Input data.
    dual : array-like
        Input data.
    result : array-like
        Input data.
    which : array-like
        Input data.
    explained : array-like
        Input data.
    below : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.9) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.9) from MVSML chapter 9."})


def cheatsheet():
    return "msm184: Numbered display equation (9.9) from MVSML chapter 9."
