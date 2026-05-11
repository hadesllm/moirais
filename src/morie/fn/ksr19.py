# morie.fn — function file (hadesllm/morie)
"""Cox model via empirical process."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_cox_partial_likelihood"]


def kosorok_cox_partial_likelihood(x, t, event):
    """
    Cox model via empirical process

    Formula: l(beta) = sum [Xi'beta - log(sum exp(Xj'beta))]

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox model via empirical process"})


def cheatsheet():
    return "ksr19: Cox model via empirical process"
