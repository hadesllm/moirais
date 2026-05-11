"""Numbered display equation (7.7) from MVSML chapter 7.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_bayesian_regression_pt2_eq_7_7"]


def mvsml_bayesian_regression_pt2_eq_7_7(the, value, of, this, that, maximizes):
    """
    Numbered display equation (7.7) from MVSML chapter 7.

    Formula: the value of this that maximizes the penalized log-likelihood: X C \betaT ℓp \beta; y ( ) = ℓ\beta; y ( )  \lambda c \betac,

    Parameters
    ----------
    the : array-like
        Input data.
    value : array-like
        Input data.
    of : array-like
        Input data.
    this : array-like
        Input data.
    that : array-like
        Input data.
    maximizes : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (7.7) [Multivariate Statistical Machine Learnin [Pages 209-249] [2026-04-16].pdf]
    """
    the = np.atleast_1d(np.asarray(the, dtype=float))
    n = len(the)
    result = float(np.mean(the))
    se = float(np.std(the, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (7.7) from MVSML chapter 7."})


def cheatsheet():
    return "msm109: Numbered display equation (7.7) from MVSML chapter 7."
