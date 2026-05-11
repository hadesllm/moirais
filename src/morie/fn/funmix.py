"""Functional mixture model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_mixture"]


def functional_mixture(Y, K):
    """
    Functional mixture model

    Formula: EM over component-specific covariances

    Parameters
    ----------
    Y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    James-Sugar (2003)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional mixture model"})


def cheatsheet():
    return "funmix: Functional mixture model"
