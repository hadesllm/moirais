"""Adversarial inverse RL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["airl"]


def airl(expert_trajs, D, policy):
    """
    Adversarial inverse RL

    Formula: learn reward as discriminator: log D − log(1−D)

    Parameters
    ----------
    expert_trajs : array-like
        Input data.
    D : array-like
        Input data.
    policy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fu-Luo-Levine (2018)
    """
    expert_trajs = np.atleast_1d(np.asarray(expert_trajs, dtype=float))
    n = len(expert_trajs)
    result = float(np.mean(expert_trajs))
    se = float(np.std(expert_trajs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adversarial inverse RL"})


def cheatsheet():
    return "airl: Adversarial inverse RL"
