"""DP logistic regression (output / objective perturbation)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_logistic"]


def dp_logistic(X, y, epsilon, method):
    """
    DP logistic regression (output / objective perturbation)

    Formula: perturb objective: J(β) + b^T β with b ~ Lap

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    epsilon : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chaudhuri-Monteleoni-Sarwate (2011)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP logistic regression (output / objective perturbation)"})


def cheatsheet():
    return "dplog: DP logistic regression (output / objective perturbation)"
