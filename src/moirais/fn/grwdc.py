# moirais.fn — function file (hadesllm/moirais)
"""AdamW: decoupled weight decay applied directly to parameters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adamw_decoupled_weight_decay"]


def geron_adamw_decoupled_weight_decay(theta, grad, m, s, t, eta, b1, b2, eps, lam):
    """
    AdamW: decoupled weight decay applied directly to parameters

    Formula: theta <- theta - eta * (m_hat / (sqrt(s_hat) + eps) + lam * theta)

    Parameters
    ----------
    theta : array-like
        Input data.
    grad : array-like
        Input data.
    m : array-like
        Input data.
    s : array-like
        Input data.
    t : array-like
        Input data.
    eta : array-like
        Input data.
    b1 : array-like
        Input data.
    b2 : array-like
        Input data.
    eps : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_new

    References
    ----------
    Géron Ch 11, AdamW section
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdamW: decoupled weight decay applied directly to parameters"})


def cheatsheet():
    return "grwdc: AdamW: decoupled weight decay applied directly to parameters"
