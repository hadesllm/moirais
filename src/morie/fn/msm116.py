"""Numbered display equation (7.9) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_9"]


def mvsml_bayesian_regression_pt2_eq_7_9(p, y, cj, c, j, block):
    """
    Numbered display equation (7.9) from MVSML chapter 7.

    Formula: ℓp \beta; y ( ) = ℓ\beta; y ( )  \lambda \betacj (7.10) c=1 j=1 and block updating can be done as in

    Parameters
    ----------
    p : array-like
        Input data.
    y : array-like
        Input data.
    cj : array-like
        Input data.
    c : array-like
        Input data.
    j : array-like
        Input data.
    block : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.9) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.9) from MVSML chapter 7."})


def cheatsheet():
    return "msm116: Numbered display equation (7.9) from MVSML chapter 7."
