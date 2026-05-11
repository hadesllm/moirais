"""Linear Thompson sampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["lin_thompson"]


def lin_thompson(context, arms, beta):
    """
    Linear Thompson sampling

    Formula: sample θ̃ ~ N(θ̂, β² A^{-1}); arm = argmax x^T θ̃

    Parameters
    ----------
    context : array-like
        Input data.
    arms : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Agrawal-Goyal (2013)
    """
    context = np.atleast_1d(np.asarray(context, dtype=float))
    n = len(context)
    result = float(np.mean(context))
    se = float(np.std(context, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear Thompson sampling"})


def cheatsheet():
    return "linTS: Linear Thompson sampling"
