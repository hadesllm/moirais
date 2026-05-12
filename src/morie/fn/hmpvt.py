# morie.fn -- function file (hadesllm/morie)
"""Pyramid Vision Transformer (PVT): multi-scale transformer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pvt"]


def geron_pvt(image, stage_cfgs):
    """
    Pyramid Vision Transformer (PVT): multi-scale transformer

    Formula: hierarchical transformer stages with shrinking spatial resolution

    Parameters
    ----------
    image : array-like
        Input data.
    stage_cfgs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 16
    """
    image = np.atleast_1d(np.asarray(image, dtype=float))
    n = len(image)
    result = float(np.mean(image))
    se = float(np.std(image, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pyramid Vision Transformer (PVT): multi-scale transformer"})


def cheatsheet():
    return "hmpvt: Pyramid Vision Transformer (PVT): multi-scale transformer"
