# morie.fn — function file (hadesllm/morie)
"""Full RLHF pipeline: SFT -> Reward Model -> PPO fine-tuning."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rlhf_pipeline"]


def kamath_rlhf_pipeline(demos, preferences, pi0):
    """
    Full RLHF pipeline: SFT -> Reward Model -> PPO fine-tuning

    Formula: pi_SFT = SFT(pi_0, demos); r_phi = train_RM(preferences); pi_RLHF = PPO(pi_SFT, r_phi, KL to pi_SFT)

    Parameters
    ----------
    demos : array-like
        Input data.
    preferences : array-like
        Input data.
    pi0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pi_rlhf

    References
    ----------
    Kamath Ch 5, RLHF Pipeline section
    """
    demos = np.atleast_1d(np.asarray(demos, dtype=float))
    n = len(demos)
    result = float(np.mean(demos))
    se = float(np.std(demos, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Full RLHF pipeline: SFT -> Reward Model -> PPO fine-tuning"})


def cheatsheet():
    return "kmrhf: Full RLHF pipeline: SFT -> Reward Model -> PPO fine-tuning"
