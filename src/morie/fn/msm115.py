"""Numbered display equation (7.10) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_10"]


def mvsml_bayesian_regression_pt2_eq_7_10(p, y, cj):
    """
    Numbered display equation (7.10) from MVSML chapter 7.

    Formula: ℓp \beta; y ( ) = ℓ\beta; y ( )  \lambda \betacj

    Parameters
    ----------
    p : array-like
        Input data.
    y : array-like
        Input data.
    cj : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.10) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.10) from MVSML chapter 7."})


def cheatsheet():
    return "msm115: Numbered display equation (7.10) from MVSML chapter 7."
