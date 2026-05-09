"""Stochastic gradient boosting subsample."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_subsampling"]


def esl_subsampling(eta):
    """
    Stochastic gradient boosting subsample

    Formula: Use eta-fraction sample at each iteration

    Parameters
    ----------
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: factor

    References
    ----------
    Hastie ESL Ch 10
    """
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    n = len(eta)
    result = float(np.mean(eta))
    se = float(np.std(eta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic gradient boosting subsample"})


def cheatsheet():
    return "eslsmp: Stochastic gradient boosting subsample"
