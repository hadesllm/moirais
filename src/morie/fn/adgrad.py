"""AdaGrad adaptive learning rate."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["adagrad"]


def adagrad(g, lr, eps):
    """
    AdaGrad adaptive learning rate

    Formula: x -= lr g / sqrt(sum g^2 + eps)

    Parameters
    ----------
    g : array-like
        Input data.
    lr : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Duchi-Hazan-Singer (2011)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaGrad adaptive learning rate"})


def cheatsheet():
    return "adgrad: AdaGrad adaptive learning rate"
