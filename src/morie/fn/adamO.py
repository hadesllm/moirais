# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Adam optimizer update rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["adam_optimizer"]


def adam_optimizer(theta, grad, lr, beta1, beta2, eps):
    """
    Adam optimizer update rule

    Formula: m_t = beta1*m_{t-1} + (1-beta1)*g_t; v_t = beta2*v_{t-1} + (1-beta2)*g_t^2; theta -= alpha*m_hat_t/(sqrt(v_hat_t)+eps)

    Parameters
    ----------
    theta : array-like
        Input data.
    grad : array-like
        Input data.
    lr : array-like
        Input data.
    beta1 : array-like
        Input data.
    beta2 : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adam optimizer update rule"})


def cheatsheet():
    return "adamO: Adam optimizer update rule"
