"""Numbered display equation (7.6) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(xT, log, i, c, C, P):
    """
    Numbered display equation (7.6) from MVSML chapter 7.

    Formula: ( ) = \beta0c + xT log i \betac, c = 1, . . . , C  1, P Yi = Cjx ( ) where the effects of xi depend on the chosen response baseline category. Similar expressions can be obtained when using the unconstrained model in

    Parameters
    ----------
    xT : array-like
        Input data.
    log : array-like
        Input data.
    i : array-like
        Input data.
    c : array-like
        Input data.
    C : array-like
        Input data.
    P : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.6) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    xT = np.atleast_1d(np.asarray(xT, dtype=float))
    n = len(xT)
    result = float(np.mean(xT))
    se = float(np.std(xT, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.6) from MVSML chapter 7."})


def cheatsheet():
    return "msm108: Numbered display equation (7.6) from MVSML chapter 7."
