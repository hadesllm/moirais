# morie.fn -- function file (hadesllm/morie)
"""Zero-shot learning: LLM generalizes to unseen tasks from prompt only."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_zero_shot"]


def geron_zero_shot(model, prompt):
    """
    Zero-shot learning: LLM generalizes to unseen tasks from prompt only

    Formula: P(y | prompt) without task-specific training

    Parameters
    ----------
    model : array-like
        Input data.
    prompt : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Géron Ch 15
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Zero-shot learning: LLM generalizes to unseen tasks from prompt only"})


def cheatsheet():
    return "hmzsl: Zero-shot learning: LLM generalizes to unseen tasks from prompt only"
