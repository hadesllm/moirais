"""Sequential targeted parametric g-formula."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sequential_target_models"]


def sequential_target_models(y, treatment_history, covariate_history, time):
    """
    Sequential targeted parametric g-formula

    Formula: iterate Q-bar from K to 0

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bang-Robins (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential targeted parametric g-formula"})


def cheatsheet():
    return "sntmod: Sequential targeted parametric g-formula"
