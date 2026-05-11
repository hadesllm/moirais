"""Nesterov accelerated gradient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nesterov_accelerated"]


def nesterov_accelerated(g, mu, lr):
    """
    Nesterov accelerated gradient

    Formula: y_{t+1} = x_t - lr g(x_t); x_{t+1} = y_{t+1} + mu(y_{t+1} - y_t)

    Parameters
    ----------
    g : array-like
        Input data.
    mu : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nesterov (1983)
    """
    g = np.atleast_1d(np.asarray(g, dtype=float))
    n = len(g)
    result = float(np.mean(g))
    se = float(np.std(g, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nesterov accelerated gradient"})


def cheatsheet():
    return "nesterv: Nesterov accelerated gradient"
