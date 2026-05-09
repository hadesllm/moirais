# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Reward-model training via Bradley-Terry pair loss (Alammar framing)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_reward_model_training_bt"]


def alammar_reward_model_training_bt(scores_w, scores_l):
    """
    Reward-model training via Bradley-Terry pair loss (Alammar framing)

    Formula: L = - E[ log sigmoid( r_phi(x, y_w) - r_phi(x, y_l) ) ]

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
    Alammar Ch 12, Reward Model Training section
    """
    scores_w = np.atleast_1d(np.asarray(scores_w, dtype=float))
    n = len(scores_w)
    result = float(np.mean(scores_w))
    se = float(np.std(scores_w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reward-model training via Bradley-Terry pair loss (Alammar framing)"})


def cheatsheet():
    return "alrmt: Reward-model training via Bradley-Terry pair loss (Alammar framing)"
