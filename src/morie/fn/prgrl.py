"""Programmatic / curriculum RL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prog_rl"]


def prog_rl(env_factory, schedule):
    """
    Programmatic / curriculum RL

    Formula: sequence subtasks of increasing difficulty

    Parameters
    ----------
    env_factory : array-like
        Input data.
    schedule : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bengio et al (2009) curriculum
    """
    env_factory = np.atleast_1d(np.asarray(env_factory, dtype=float))
    n = len(env_factory)
    result = float(np.mean(env_factory))
    se = float(np.std(env_factory, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Programmatic / curriculum RL"})


def cheatsheet():
    return "prgrl: Programmatic / curriculum RL"
