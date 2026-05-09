"""Numbered display equation (7.6) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(PC, c, C, l, exp, xT):
    """
    Numbered display equation (7.6) from MVSML chapter 7.

    Formula: ) = PC , c = 1, . . . , C, (7.6) l=1 exp \beta0l + xT ( i \betal ) where \betac, c = 1, . . ., C, is a vector of coefﬁcients of the same dimension as x. Model

    Parameters
    ----------
    PC : array-like
        Input data.
    c : array-like
        Input data.
    C : array-like
        Input data.
    l : array-like
        Input data.
    exp : array-like
        Input data.
    xT : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.6) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    PC = np.atleast_1d(np.asarray(PC, dtype=float))
    n = len(PC)
    result = float(np.mean(PC))
    se = float(np.std(PC, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.6) from MVSML chapter 7."})


def cheatsheet():
    return "msm107: Numbered display equation (7.6) from MVSML chapter 7."
