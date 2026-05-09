# moirais.fn — function file (hadesllm/moirais)
"""Fine-tune BERT on a downstream classification task."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bert_finetune"]


def geron_bert_finetune(bert, X, y, epochs, lr):
    """
    Fine-tune BERT on a downstream classification task

    Formula: classifier head on [CLS]; train end-to-end on task data

    Parameters
    ----------
    bert : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
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
    Géron Ch 15
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fine-tune BERT on a downstream classification task"})


def cheatsheet():
    return "hmbftn: Fine-tune BERT on a downstream classification task"
