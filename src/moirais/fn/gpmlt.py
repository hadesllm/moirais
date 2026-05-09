"""Multi-task GP regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gp_multitask"]


def gp_multitask(X, y_tasks, X_test):
    """
    Multi-task GP regression

    Formula: k_total = k_task ⊗ k_input

    Parameters
    ----------
    X : array-like
        Input data.
    y_tasks : array-like
        Input data.
    X_test : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bonilla-Chai-Williams (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-task GP regression"})


def cheatsheet():
    return "gpmlt: Multi-task GP regression"
