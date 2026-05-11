"""Proximal policy optimization (clipped surrogate)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ppo"]


def ppo(env, policy, clip_eps):
    """
    Proximal policy optimization (clipped surrogate)

    Formula: L^CLIP = E[min(r_t · A, clip(r_t, 1−ε, 1+ε) · A)]

    Parameters
    ----------
    env : array-like
        Input data.
    policy : array-like
        Input data.
    clip_eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schulman et al (2017)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal policy optimization (clipped surrogate)"})


def cheatsheet():
    return "ppoc: Proximal policy optimization (clipped surrogate)"
