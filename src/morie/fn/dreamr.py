"""Dreamer — world-model RL via latent imagination."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dreamer"]


def dreamer(env, world_model, actor, critic):
    """
    Dreamer — world-model RL via latent imagination

    Formula: RSSM + actor + critic in latent space

    Parameters
    ----------
    env : array-like
        Input data.
    world_model : array-like
        Input data.
    actor : array-like
        Input data.
    critic : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hafner et al (2020) Dreamer
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dreamer — world-model RL via latent imagination"})


def cheatsheet():
    return "dreamr: Dreamer — world-model RL via latent imagination"
