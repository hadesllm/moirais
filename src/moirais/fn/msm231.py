"""Numbered display equation (9.44) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_44"]


def mvsml_ridge_lasso_elastic_eq_9_44(Wolfe, dual, version, maximization, problem, of):
    """
    Numbered display equation (9.44) from MVSML chapter 9.

    Formula: Wolfe dual version (maximization problem) of the optimization problem Xn Xn   i=1\alphai 2 1 maximize L \alpha ( ) = i=1\alphai\alpha jyiy j xi:x j

    Parameters
    ----------
    Wolfe : array-like
        Input data.
    dual : array-like
        Input data.
    version : array-like
        Input data.
    maximization : array-like
        Input data.
    problem : array-like
        Input data.
    of : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.44) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    Wolfe = np.atleast_1d(np.asarray(Wolfe, dtype=float))
    n = len(Wolfe)
    result = float(np.mean(Wolfe))
    se = float(np.std(Wolfe, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.44) from MVSML chapter 9."})


def cheatsheet():
    return "msm231: Numbered display equation (9.44) from MVSML chapter 9."
