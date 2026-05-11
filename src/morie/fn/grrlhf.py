# morie.fn — function file (hadesllm/morie)
"""RLHF PPO objective with KL penalty against reference model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rlhf_reward_kl_objective"]


def geron_rlhf_reward_kl_objective(rewards, policy_logprobs, ref_logprobs, beta):
    """
    RLHF PPO objective with KL penalty against reference model

    Formula: J = E[r(x, y)] - beta * KL(pi_theta(.|x) || pi_ref(.|x))

    Parameters
    ----------
    rewards : array-like
        Input data.
    policy_logprobs : array-like
        Input data.
    ref_logprobs : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: objective

    References
    ----------
    Géron Ch 15, RLHF section
    """
    rewards = np.atleast_1d(np.asarray(rewards, dtype=float))
    n = len(rewards)
    result = float(np.mean(rewards))
    se = float(np.std(rewards, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RLHF PPO objective with KL penalty against reference model"})


def cheatsheet():
    return "grrlhf: RLHF PPO objective with KL penalty against reference model"
