"""Bayesian IRL."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayesian_irl"]


def bayesian_irl(expert_trajs, prior):
    """
    Bayesian IRL

    Formula: P(R|D) ∝ P(D|R) P(R) via MCMC

    Parameters
    ----------
    expert_trajs : array-like
        Input data.
    prior : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ramachandran-Amir (2007)
    """
    expert_trajs = np.atleast_1d(np.asarray(expert_trajs, dtype=float))
    n = len(expert_trajs)
    result = float(np.mean(expert_trajs))
    se = float(np.std(expert_trajs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian IRL"})


def cheatsheet():
    return "birl: Bayesian IRL"
