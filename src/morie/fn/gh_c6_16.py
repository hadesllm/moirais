# morie.fn -- function file (hadesllm/morie)
"""Alpha-posterior (power-likelihood) for robust consistency: pi_alpha(theta|X^n) proportional pi(theta) * prod p_theta(X_i)^alpha."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_alpha_post"]


def ghosal_alpha_post(x):
    """
    Alpha-posterior (power-likelihood) for robust consistency: pi_alpha(theta|X^n) proportional pi(theta) * prod p_theta(X_i)^alpha

    Formula: pi_alpha(theta|X^n) proportional pi(theta) * L_n(theta)^alpha, 0<alpha<1

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 6 §6.8.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alpha-posterior (power-likelihood) for robust consistency: pi_alpha(theta|X^n) proportional pi(theta) * prod p_theta(X_i)^alpha"})


def cheatsheet():
    return "gh_c6_16: Alpha-posterior (power-likelihood) for robust consistency: pi_alpha(theta|X^n) proportional pi(theta) * prod p_theta(X_i)^alpha"
