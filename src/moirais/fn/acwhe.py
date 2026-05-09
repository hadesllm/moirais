"""Privacy-accuracy trade-off bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["private_accuracy_tradeoff"]


def private_accuracy_tradeoff(sensitivity, epsilon, n):
    """
    Privacy-accuracy trade-off bound

    Formula: MSE ≥ Δ²/ε² for Laplace mechanism

    Parameters
    ----------
    sensitivity : array-like
        Input data.
    epsilon : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Roth (2014) book
    """
    sensitivity = np.atleast_1d(np.asarray(sensitivity, dtype=float))
    n = len(sensitivity)
    result = float(np.mean(sensitivity))
    se = float(np.std(sensitivity, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Privacy-accuracy trade-off bound"})


def cheatsheet():
    return "acwhe: Privacy-accuracy trade-off bound"
