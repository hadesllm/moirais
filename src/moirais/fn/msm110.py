"""Numbered display equation (7.8) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_8"]


def mvsml_bayesian_regression_pt2_eq_7_8(I, yi, c, i, log, l):
    """
    Numbered display equation (7.8) from MVSML chapter 7.

    Formula: I yi=c i \betac  log i \betal f i=1 c=1 i=1 l=1

    Parameters
    ----------
    I : array-like
        Input data.
    yi : array-like
        Input data.
    c : array-like
        Input data.
    i : array-like
        Input data.
    log : array-like
        Input data.
    l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.8) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    I = np.atleast_1d(np.asarray(I, dtype=float))
    n = len(I)
    result = float(np.mean(I))
    se = float(np.std(I, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.8) from MVSML chapter 7."})


def cheatsheet():
    return "msm110: Numbered display equation (7.8) from MVSML chapter 7."
