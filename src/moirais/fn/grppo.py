# moirais.fn — function file (hadesllm/moirais)
"""Proximal Policy Optimization clipped surrogate objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_ppo_clipped_objective"]


def geron_ppo_clipped_objective(ratios, advantages, eps):
    """
    Proximal Policy Optimization clipped surrogate objective

    Formula: L = E_t [ min( r_t(theta)*A_t, clip(r_t(theta), 1-eps, 1+eps)*A_t ) ]; r_t = pi_theta/pi_old

    Parameters
    ----------
    ratios : array-like
        Input data.
    advantages : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: objective

    References
    ----------
    Géron Ch 19, PPO section
    """
    ratios = np.atleast_1d(np.asarray(ratios, dtype=float))
    n = len(ratios)
    result = float(np.mean(ratios))
    se = float(np.std(ratios, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal Policy Optimization clipped surrogate objective"})


def cheatsheet():
    return "grppo: Proximal Policy Optimization clipped surrogate objective"
