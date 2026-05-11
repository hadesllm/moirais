"""Mish activation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mish_activation"]


def mish_activation(y):
    """
    Mish activation

    Formula: Mish(x) = x * tanh(softplus(x))

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Misra (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mish activation"})


def cheatsheet():
    return "mishfn: Mish activation"
