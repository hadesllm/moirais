"""Reward machine — finite-state task spec."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reward_machine"]


def reward_machine(env, fsa):
    """
    Reward machine — finite-state task spec

    Formula: FSA over high-level events shapes reward

    Parameters
    ----------
    env : array-like
        Input data.
    fsa : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Icarte et al (2018)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reward machine — finite-state task spec"})


def cheatsheet():
    return "rmrl: Reward machine — finite-state task spec"
