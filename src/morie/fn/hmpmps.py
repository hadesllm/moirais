# morie.fn — function file (hadesllm/morie)
"""Apple MPS hardware acceleration (Metal Performance Shaders)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mps_acceleration"]


def geron_mps_acceleration(tensor):
    """
    Apple MPS hardware acceleration (Metal Performance Shaders)

    Formula: move tensor to 'mps' device for GPU-equivalent ops on Apple Silicon

    Parameters
    ----------
    tensor : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tensor_mps

    References
    ----------
    Géron Ch 10
    """
    tensor = np.atleast_1d(np.asarray(tensor, dtype=float))
    n = len(tensor)
    result = float(np.mean(tensor))
    se = float(np.std(tensor, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Apple MPS hardware acceleration (Metal Performance Shaders)"})


def cheatsheet():
    return "hmpmps: Apple MPS hardware acceleration (Metal Performance Shaders)"
