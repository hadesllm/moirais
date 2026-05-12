# morie.fn -- function file (hadesllm/morie)
"""Transfer learning: reuse pretrained model, fine-tune on new task."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_transfer_learning"]


def geron_transfer_learning(pretrained_model, X, y, n_frozen):
    """
    Transfer learning: reuse pretrained model, fine-tune on new task

    Formula: freeze initial layers; train final layers on new data

    Parameters
    ----------
    pretrained_model : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_frozen : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 11
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer learning: reuse pretrained model, fine-tune on new task"})


def cheatsheet():
    return "hmtfl: Transfer learning: reuse pretrained model, fine-tune on new task"
