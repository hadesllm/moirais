# moirais.fn — function file (hadesllm/moirais)
"""Reward-model training loss over preference pairs (BT negative log-likelihood)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_reward_model_training_loss"]


def kamath_reward_model_training_loss(scores_w, scores_l):
    """
    Reward-model training loss over preference pairs (BT negative log-likelihood)

    Formula: L_RM = - E_{(x, y_w, y_l)} [ log sigmoid( r_phi(x, y_w) - r_phi(x, y_l) ) ]

    Parameters
    ----------
    scores_w : array-like
        Input data.
    scores_l : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Kamath Ch 5, Reward Model Training section
    """
    scores_w = np.atleast_1d(np.asarray(scores_w, dtype=float))
    n = len(scores_w)
    result = float(np.mean(scores_w))
    se = float(np.std(scores_w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reward-model training loss over preference pairs (BT negative log-likelihood)"})


def cheatsheet():
    return "kmrmloss: Reward-model training loss over preference pairs (BT negative log-likelihood)"
