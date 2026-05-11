"""Minimum description length."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_mdl"]


def esl_mdl(loglik, theta):
    """
    Minimum description length

    Formula: MDL = -log P(y|X,theta) - log P(theta)

    Parameters
    ----------
    loglik : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    loglik = np.atleast_1d(np.asarray(loglik, dtype=float))
    n = len(loglik)
    result = float(np.mean(loglik))
    se = float(np.std(loglik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Minimum description length"})


def cheatsheet():
    return "eslmdl: Minimum description length"
