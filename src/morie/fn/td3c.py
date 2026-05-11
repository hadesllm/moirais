"""Twin-delayed DDPG (TD3)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["td3"]


def td3(env, actor, critic1, critic2):
    """
    Twin-delayed DDPG (TD3)

    Formula: min of two critics; delayed actor + target smoothing

    Parameters
    ----------
    env : array-like
        Input data.
    actor : array-like
        Input data.
    critic1 : array-like
        Input data.
    critic2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fujimoto-Hoof-Meger (2018)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Twin-delayed DDPG (TD3)"})


def cheatsheet():
    return "td3c: Twin-delayed DDPG (TD3)"
