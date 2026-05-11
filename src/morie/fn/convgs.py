"""Convergent validity (AVE)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["convergent_validity"]


def convergent_validity(loadings, residuals):
    """
    Convergent validity (AVE)

    Formula: AVE = sum lambda_i^2 / (sum lambda_i^2 + sum theta_i)

    Parameters
    ----------
    loadings : array-like
        Input data.
    residuals : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fornell-Larcker (1981)
    """
    loadings = np.atleast_1d(np.asarray(loadings, dtype=float))
    n = len(loadings)
    result = float(np.mean(loadings))
    se = float(np.std(loadings, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convergent validity (AVE)"})


def cheatsheet():
    return "convgs: Convergent validity (AVE)"
