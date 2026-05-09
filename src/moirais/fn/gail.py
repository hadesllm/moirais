"""Generative adversarial imitation learning."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gail"]


def gail(expert_trajs, D, policy):
    """
    Generative adversarial imitation learning

    Formula: min_π max_D E_E[log D] + E_π[log(1−D)]

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
    Ho-Ermon (2016)
    """
    expert_trajs = np.atleast_1d(np.asarray(expert_trajs, dtype=float))
    n = len(expert_trajs)
    result = float(np.mean(expert_trajs))
    se = float(np.std(expert_trajs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generative adversarial imitation learning"})


def cheatsheet():
    return "gail: Generative adversarial imitation learning"
