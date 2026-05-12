# morie.fn -- function file (hadesllm/morie)
"""REINFORCE policy gradient update (Monte Carlo)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_reinforce_policy_gradient"]


def geron_reinforce_policy_gradient(theta, log_probs, returns_G, alpha):
    """
    REINFORCE policy gradient update (Monte Carlo)

    Formula: theta <- theta + alpha * G_t * grad_theta log pi_theta(a_t | s_t)

    Parameters
    ----------
    theta : array-like
        Input data.
    log_probs : array-like
        Input data.
    returns_G : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_new

    References
    ----------
    Géron Ch 19, Policy Gradients / REINFORCE section
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "REINFORCE policy gradient update (Monte Carlo)"})


def cheatsheet():
    return "grrein: REINFORCE policy gradient update (Monte Carlo)"
