"""Numbered display equation (7.9) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_9"]


def mvsml_bayesian_regression_pt2_eq_7_9(of, e, That, the, update, block):
    """
    Numbered display equation (7.9) from MVSML chapter 7.

    Formula: 227 of \beta (e\beta). That is, the update of block c is achieved by maximizing the following function with respect to \beta0c and \betac: ) = ℓ )  \lambda\betaT f c \beta0c, \betac ( c \beta; y ( c \betac,

    Parameters
    ----------
    of : array-like
        Input data.
    e : array-like
        Input data.
    That : array-like
        Input data.
    the : array-like
        Input data.
    update : array-like
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
    of = np.atleast_1d(np.asarray(of, dtype=float))
    n = len(of)
    result = float(np.mean(of))
    se = float(np.std(of, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.9) from MVSML chapter 7."})


def cheatsheet():
    return "msm112: Numbered display equation (7.9) from MVSML chapter 7."
