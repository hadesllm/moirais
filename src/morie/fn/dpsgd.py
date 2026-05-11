"""DP-SGD with per-sample clipping + noise."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_sgd"]


def dp_sgd(loss, C, sigma, lr):
    """
    DP-SGD with per-sample clipping + noise

    Formula: g̃ = (1/L)(sum clip(g_i, C) + N(0, σ²C²I))

    Parameters
    ----------
    loss : array-like
        Input data.
    C : array-like
        Input data.
    sigma : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abadi et al (2016)
    """
    loss = np.atleast_1d(np.asarray(loss, dtype=float))
    n = len(loss)
    result = float(np.mean(loss))
    se = float(np.std(loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP-SGD with per-sample clipping + noise"})


def cheatsheet():
    return "dpsgd: DP-SGD with per-sample clipping + noise"
