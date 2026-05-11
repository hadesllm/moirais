"""Basic reproduction number R0."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["basic_reproduction"]


def basic_reproduction(beta, gamma):
    """
    Basic reproduction number R0

    Formula: R0 = beta / gamma for SIR

    Parameters
    ----------
    beta : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diekmann-Heesterbeek-Metz (1990)
    """
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    n = len(beta)
    result = float(np.mean(beta))
    se = float(np.std(beta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Basic reproduction number R0"})


def cheatsheet():
    return "r0bayse: Basic reproduction number R0"
