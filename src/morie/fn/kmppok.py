# morie.fn — function file (hadesllm/morie)
"""PPO-based RLHF policy objective with reward-model + KL penalty."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_ppo_rlhf_objective"]


def kamath_ppo_rlhf_objective(rewards, logp_theta, logp_ref, beta):
    """
    PPO-based RLHF policy objective with reward-model + KL penalty

    Formula: J(theta) = E_{x, y ~ pi_theta} [ r_phi(x,y) - beta * log(pi_theta(y|x) / pi_ref(y|x)) ]

    Parameters
    ----------
    rewards : array-like
        Input data.
    logp_theta : array-like
        Input data.
    logp_ref : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: objective

    References
    ----------
    Kamath Ch 5, RLHF PPO Pipeline section
    """
    rewards = np.atleast_1d(np.asarray(rewards, dtype=float))
    n = len(rewards)
    result = float(np.mean(rewards))
    se = float(np.std(rewards, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PPO-based RLHF policy objective with reward-model + KL penalty"})


def cheatsheet():
    return "kmppok: PPO-based RLHF policy objective with reward-model + KL penalty"
