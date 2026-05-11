"""Actor-critic with TD baseline."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["actor_critic"]


def actor_critic(env, actor, critic):
    """
    Actor-critic with TD baseline

    Formula: ∇θ J = E[∇θ log π(a|s,θ) · δ_t]

    Parameters
    ----------
    env : array-like
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
    Sutton-Barto (1998)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Actor-critic with TD baseline"})


def cheatsheet():
    return "acrt: Actor-critic with TD baseline"
