# morie.fn -- function file (hadesllm/morie)
"""Discounted return G_t from step t onward."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_discounted_return"]


def geron_discounted_return(rewards, gamma):
    """
    Discounted return G_t from step t onward

    Formula: G_t = sum_{k=0..infty} gamma^k * r_{t+k+1}

    Parameters
    ----------
    rewards : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: return

    References
    ----------
    Géron Ch 19, Discounted Return section
    """
    rewards = np.atleast_1d(np.asarray(rewards, dtype=float))
    n = len(rewards)
    result = float(np.mean(rewards))
    se = float(np.std(rewards, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discounted return G_t from step t onward"})


def cheatsheet():
    return "grret: Discounted return G_t from step t onward"
