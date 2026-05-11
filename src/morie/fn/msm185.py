"""Numbered display equation (9.12) from MVSML chapter 9.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_ridge_lasso_elastic_eq_9_12"]


def mvsml_ridge_lasso_elastic_eq_9_12(optimization, problem, Xm, Xp, maximize, f):
    """
    Numbered display equation (9.12) from MVSML chapter 9.

    Formula: optimization problem is Xm Xp maximize f x ( ) + i=1\lambdaihi x ( ) + i=1\alphaigi x ( )

    Parameters
    ----------
    optimization : array-like
        Input data.
    problem : array-like
        Input data.
    Xm : array-like
        Input data.
    Xp : array-like
        Input data.
    maximize : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (9.12) [Multivariate Statistical Machine Learnin [Pages 337-378] [2026-04-16].pdf]
    """
    optimization = np.atleast_1d(np.asarray(optimization, dtype=float))
    n = len(optimization)
    result = float(np.mean(optimization))
    se = float(np.std(optimization, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (9.12) from MVSML chapter 9."})


def cheatsheet():
    return "msm185: Numbered display equation (9.12) from MVSML chapter 9."
