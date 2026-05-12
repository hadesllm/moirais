# morie.fn -- function file (hadesllm/morie)
"""Prioritized experience replay importance-sampling weight."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_prioritized_experience_weight"]


def geron_prioritized_experience_weight(priorities, N, alpha, beta):
    """
    Prioritized experience replay importance-sampling weight

    Formula: w_i = (N * P(i))^{-beta} / max_j w_j;  P(i) prop (|delta_i| + eps)^alpha

    Parameters
    ----------
    priorities : array-like
        Input data.
    N : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights

    References
    ----------
    Géron Ch 19, Prioritized Experience Replay section
    """
    priorities = np.atleast_1d(np.asarray(priorities, dtype=float))
    n = len(priorities)
    result = float(np.mean(priorities))
    se = float(np.std(priorities, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prioritized experience replay importance-sampling weight"})


def cheatsheet():
    return "grpex: Prioritized experience replay importance-sampling weight"
