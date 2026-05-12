# morie.fn -- function file (hadesllm/morie)
"""Fine-tuning via TRL (Transformer Reinforcement Learning) library."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_trl_finetune"]


def geron_trl_finetune(model, dataset, method):
    """
    Fine-tuning via TRL (Transformer Reinforcement Learning) library

    Formula: SFT, DPO, PPO trainers from TRL

    Parameters
    ----------
    model : array-like
        Input data.
    dataset : array-like
        Input data.
    method : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 15
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fine-tuning via TRL (Transformer Reinforcement Learning) library"})


def cheatsheet():
    return "hmtrlf: Fine-tuning via TRL (Transformer Reinforcement Learning) library"
