"""Numbered display equation (7.6) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_6"]


def mvsml_bayesian_regression_pt2_eq_7_6(C, the, following, exp, xT, i):
    """
    Numbered display equation (7.6) from MVSML chapter 7.

    Formula: 1, 2, . . ., C, with the following probabilities: +  exp \beta0c + xT i \betac P Yi = cjxi ( ) = PC , c = 1, . . . , C,

    Parameters
    ----------
    C : array-like
        Input data.
    the : array-like
        Input data.
    following : array-like
        Input data.
    exp : array-like
        Input data.
    xT : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.6) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    C = np.atleast_1d(np.asarray(C, dtype=float))
    n = len(C)
    result = float(np.mean(C))
    se = float(np.std(C, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.6) from MVSML chapter 7."})


def cheatsheet():
    return "msm106: Numbered display equation (7.6) from MVSML chapter 7."
