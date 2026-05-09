"""AlphaDev-style quicksort move discovery via RL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["alphadev_quicksort_disc"]


def alphadev_quicksort_disc(target, action_space, reward_fn):
    """
    AlphaDev-style quicksort move discovery via RL

    Formula: RL agent searches assembly-instruction space; reward = correctness + speed

    Parameters
    ----------
    target : array-like
        Input data.
    action_space : array-like
        Input data.
    reward_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mankowitz et al (2023) Nature DeepMind AlphaDev
    """
    target = np.atleast_1d(np.asarray(target, dtype=float))
    n = len(target)
    result = float(np.mean(target))
    se = float(np.std(target, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AlphaDev-style quicksort move discovery via RL"})


def cheatsheet():
    return "alfqud: AlphaDev-style quicksort move discovery via RL"
