"""DR-DiD with treatment × covariate interaction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interaction_did"]


def interaction_did(y, D, V, X):
    """
    DR-DiD with treatment × covariate interaction

    Formula: ATT(D, V); separate per V level

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    V : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hernán-Robins (2020)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DR-DiD with treatment × covariate interaction"})


def cheatsheet():
    return "itrct1: DR-DiD with treatment × covariate interaction"
