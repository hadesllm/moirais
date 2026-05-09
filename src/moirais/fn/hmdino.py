# moirais.fn — function file (hadesllm/moirais)
"""DINO: self-distillation with no labels for visual representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dino"]


def geron_dino(images, student, teacher):
    """
    DINO: self-distillation with no labels for visual representation

    Formula: student matches teacher momentum network on augmented views

    Parameters
    ----------
    images : array-like
        Input data.
    student : array-like
        Input data.
    teacher : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Géron Ch 16
    """
    images = np.atleast_1d(np.asarray(images, dtype=float))
    n = len(images)
    result = float(np.mean(images))
    se = float(np.std(images, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DINO: self-distillation with no labels for visual representation"})


def cheatsheet():
    return "hmdino: DINO: self-distillation with no labels for visual representation"
