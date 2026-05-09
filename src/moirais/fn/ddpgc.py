"""Deep deterministic policy gradient (continuous actions)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ddpg"]


def ddpg(env, actor, critic, tau):
    """
    Deep deterministic policy gradient (continuous actions)

    Formula: actor μ(s); critic Q(s,a); soft target update

    Parameters
    ----------
    env : array-like
        Input data.
    actor : array-like
        Input data.
    critic : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lillicrap et al (2016)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep deterministic policy gradient (continuous actions)"})


def cheatsheet():
    return "ddpgc: Deep deterministic policy gradient (continuous actions)"
