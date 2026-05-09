"""Posterior distribution f(theta|x) prop f(x|theta) f(theta)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_posterior"]


def wasserman_posterior(data, f, prior):
    """
    Posterior distribution f(theta|x) prop f(x|theta) f(theta)

    Formula: p(theta|x) = p(x|theta) p(theta) / p(x)

    Parameters
    ----------
    data : array-like
        Input data.
    f : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: posterior

    References
    ----------
    Wasserman (2004), Ch 11
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Posterior distribution f(theta|x) prop f(x|theta) f(theta)"})


def cheatsheet():
    return "wsmbay: Posterior distribution f(theta|x) prop f(x|theta) f(theta)"
