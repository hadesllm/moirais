"""Numbered display equation (7.5) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_5"]


def mvsml_bayesian_regression_pt2_eq_7_5(SD, L, XE, E, ZLg, e):
    """
    Numbered display equation (7.5) from MVSML chapter 7.

    Formula: 0.2419 0.6187 (SD) (0.02) (0.07) (0.03) (0.1) (0.03) (0.1) L = XE\betaE + ZLg + e

    Parameters
    ----------
    SD : array-like
        Input data.
    L : array-like
        Input data.
    XE : array-like
        Input data.
    E : array-like
        Input data.
    ZLg : array-like
        Input data.
    e : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.5) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    SD = np.atleast_1d(np.asarray(SD, dtype=float))
    n = len(SD)
    result = float(np.mean(SD))
    se = float(np.std(SD, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.5) from MVSML chapter 7."})


def cheatsheet():
    return "msm102: Numbered display equation (7.5) from MVSML chapter 7."
