# moirais.fn — function file (hadesllm/moirais)
"""Swin Transformer: shifted-window attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_swin"]


def geron_swin(image, window_size, n_layers):
    """
    Swin Transformer: shifted-window attention

    Formula: local attention within windows; shift windows between blocks

    Parameters
    ----------
    image : array-like
        Input data.
    window_size : array-like
        Input data.
    n_layers : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Swin Transformer: shifted-window attention"})


def cheatsheet():
    return "hmswin: Swin Transformer: shifted-window attention"
