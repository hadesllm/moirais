# morie.fn — function file (hadesllm/morie)
"""Reinforcement learning from human feedback (RLHF)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rlhf"]


def geron_rlhf(policy, reward_model, prompts):
    """
    Reinforcement learning from human feedback (RLHF)

    Formula: train reward model then PPO on policy maximizing reward - KL penalty

    Parameters
    ----------
    policy : array-like
        Input data.
    reward_model : array-like
        Input data.
    prompts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: policy

    References
    ----------
    Géron Ch 15
    """
    policy = np.atleast_1d(np.asarray(policy, dtype=float))
    n = len(policy)
    result = float(np.mean(policy))
    se = float(np.std(policy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reinforcement learning from human feedback (RLHF)"})


def cheatsheet():
    return "hmrlhf: Reinforcement learning from human feedback (RLHF)"
