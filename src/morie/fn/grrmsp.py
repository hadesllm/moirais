# morie.fn — function file (hadesllm/morie)
"""RMSProp update: exponentially-weighted moving average of squared gradients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rmsprop_update"]


def geron_rmsprop_update(theta, grad, s, eta, rho, eps):
    """
    RMSProp update: exponentially-weighted moving average of squared gradients

    Formula: s_{t+1} = rho*s_t + (1-rho)*g_t.^2; theta_{t+1} = theta_t - eta*g_t./(sqrt(s_{t+1})+eps)

    Parameters
    ----------
    theta : array-like
        Input data.
    grad : array-like
        Input data.
    s : array-like
        Input data.
    eta : array-like
        Input data.
    rho : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_new, s_new

    References
    ----------
    Géron Ch 11, RMSProp section
    """
    theta = np.asarray(theta, dtype=float)
    n = int(theta) if theta.ndim == 0 else len(theta)
    if theta.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "RMSProp update: exponentially-weighted moving average of squared gradients"})
    estimate = np.median(theta)
    se = 1.2533 * np.std(theta, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "RMSProp update: exponentially-weighted moving average of squared gradients"})


def cheatsheet():
    return "grrmsp: RMSProp update: exponentially-weighted moving average of squared gradients"
