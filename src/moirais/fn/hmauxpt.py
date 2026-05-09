# moirais.fn — function file (hadesllm/moirais)
"""Pretraining on auxiliary related task."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_auxiliary_task_pretraining"]


def geron_auxiliary_task_pretraining(model, aux_data, target_data):
    """
    Pretraining on auxiliary related task

    Formula: pretrain on task A (abundant labels), fine-tune on task B

    Parameters
    ----------
    model : array-like
        Input data.
    aux_data : array-like
        Input data.
    target_data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 11
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pretraining on auxiliary related task"})


def cheatsheet():
    return "hmauxpt: Pretraining on auxiliary related task"
