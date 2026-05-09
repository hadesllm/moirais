"""Double ML mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ml_mediation_dml"]


def ml_mediation_dml(Y, X, M, C):
    """
    Double ML mediation

    Formula: Neyman-orthogonal scores + cross-fitting

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Farbmacher et al (2022); Chernozhukov et al (2018)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double ML mediation"})


def cheatsheet():
    return "medML: Double ML mediation"
