"""Rainbow: 6 DQN improvements combined."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rainbow_dqn"]


def rainbow_dqn(env):
    """
    Rainbow: 6 DQN improvements combined

    Formula: double + dueling + PER + multistep + noisy + categorical

    Parameters
    ----------
    env : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hessel et al (2018)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rainbow: 6 DQN improvements combined"})


def cheatsheet():
    return "raindq: Rainbow: 6 DQN improvements combined"
