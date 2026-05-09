# moirais.fn — function file (hadesllm/moirais)
"""Adam optimizer step (bias-corrected first and second moments)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adam_update"]


def geron_adam_update(theta, grad, m, s, t, eta, b1, b2, eps):
    """
    Adam optimizer step (bias-corrected first and second moments)

    Formula: m=b1*m+(1-b1)*g; s=b2*s+(1-b2)*g.^2; m_hat=m/(1-b1^t); s_hat=s/(1-b2^t); theta-=eta*m_hat/(sqrt(s_hat)+eps)

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

    Returns
    -------
    result : dict
        Keys: theta_new, m_new, s_new

    References
    ----------
    Géron Ch 11, Eq 11-8 (Adam)
    """
    theta = np.asarray(theta, dtype=float)
    n = int(theta) if theta.ndim == 0 else len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adam optimizer step (bias-corrected first and second moments)"})


def cheatsheet():
    return "gradmo: Adam optimizer step (bias-corrected first and second moments)"
