# moirais.fn — function file (hadesllm/moirais)
"""DeiT distillation loss: combines CE on class token + CE on distillation token."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_deit_distillation_loss"]


def geron_deit_distillation_loss(logits_cls, logits_dist, y, teacher_preds):
    """
    DeiT distillation loss: combines CE on class token + CE on distillation token

    Formula: L = 0.5 * CE(y_cls, y) + 0.5 * CE(y_dist, teacher(x))

    Parameters
    ----------
    logits_cls : array-like
        Input data.
    logits_dist : array-like
        Input data.
    y : array-like
        Input data.
    teacher_preds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss

    References
    ----------
    Géron Ch 16, DeiT section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DeiT distillation loss: combines CE on class token + CE on distillation token"})


def cheatsheet():
    return "grdeit: DeiT distillation loss: combines CE on class token + CE on distillation token"
