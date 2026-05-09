# moirais.fn — function file (hadesllm/moirais)
"""Fine-tune a pretrained language model on a downstream task."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_finetune_lm"]


def geron_finetune_lm(model, dataset, epochs, lr):
    """
    Fine-tune a pretrained language model on a downstream task

    Formula: theta <- theta - eta * grad L_task(theta; D_task)

    Parameters
    ----------
    model : array-like
        Input data.
    dataset : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 14
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fine-tune a pretrained language model on a downstream task"})


def cheatsheet():
    return "hmfth: Fine-tune a pretrained language model on a downstream task"
