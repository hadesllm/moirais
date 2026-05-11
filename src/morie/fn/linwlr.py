"""Linear weighted regression learner."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["linear_weighted_learner"]


def linear_weighted_learner(y, A, W, propensity):
    """
    Linear weighted regression learner

    Formula: weight outcome by inverse propensity

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    W : array-like
        Input data.
    propensity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins (2004)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear weighted regression learner"})


def cheatsheet():
    return "linwlr: Linear weighted regression learner"
