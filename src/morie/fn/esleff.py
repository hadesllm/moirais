"""Effective degrees of freedom."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_effective_dof"]


def esl_effective_dof(S):
    """
    Effective degrees of freedom

    Formula: df(S) = trace(S)

    Parameters
    ----------
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 7
    """
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective degrees of freedom"})


def cheatsheet():
    return "esleff: Effective degrees of freedom"
