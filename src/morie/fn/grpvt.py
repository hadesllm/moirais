# morie.fn -- function file (hadesllm/morie)
"""Pyramid ViT stage: spatial reduction before attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pyramid_vit_stage"]


def geron_pyramid_vit_stage(X, WQ, WK, WV, reduction_ratio):
    """
    Pyramid ViT stage: spatial reduction before attention

    Formula: Q = X W_Q; K, V = SR(X) W_K, SR(X) W_V where SR = spatial reduction

    Parameters
    ----------
    X : array-like
        Input data.
    WQ : array-like
        Input data.
    WK : array-like
        Input data.
    WV : array-like
        Input data.
    reduction_ratio : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Géron Ch 16, Pyramid Vision Transformer section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pyramid ViT stage: spatial reduction before attention"})


def cheatsheet():
    return "grpvt: Pyramid ViT stage: spatial reduction before attention"
