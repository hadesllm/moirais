# morie.fn — function file (hadesllm/morie)
"""RMSProp optimizer update rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rmsprop_optimizer"]


def rmsprop_optimizer(theta, grad, lr, rho, eps):
    """
    RMSProp optimizer update rule

    Formula: E[g^2]_t = rho*E[g^2]_{t-1} + (1-rho)*g_t^2; theta -= (alpha / sqrt(E[g^2]_t + eps)) * g_t

    Parameters
    ----------
    theta : array-like
        Input data.
    grad : array-like
        Input data.
    lr : array-like
        Input data.
    rho : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'theta_new': 'array'}

    References
    ----------
    Montesinos Lopez Ch 10
    """
    theta = np.asarray(theta, dtype=float)
    n = int(theta) if theta.ndim == 0 else len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RMSProp optimizer update rule"})


def cheatsheet():
    return "rmspO: RMSProp optimizer update rule"
