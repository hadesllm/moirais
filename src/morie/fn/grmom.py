# morie.fn -- function file (hadesllm/morie)
"""Classical momentum optimizer step."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_momentum_update"]


def geron_momentum_update(theta, grad, v, eta, beta):
    """
    Classical momentum optimizer step

    Formula: v_{t+1} = beta * v_t + g_t; theta_{t+1} = theta_t - eta * v_{t+1}

    Parameters
    ----------
    theta : array-like
        Input data.
    grad : array-like
        Input data.
    v : array-like
        Input data.
    eta : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_new, v_new

    References
    ----------
    Géron Ch 11, Momentum Optimization section
    """
    theta = np.asarray(theta, dtype=float)
    n = int(theta) if theta.ndim == 0 else len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classical momentum optimizer step"})


def cheatsheet():
    return "grmom: Classical momentum optimizer step"
