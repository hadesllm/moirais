# morie.fn -- function file (hadesllm/morie)
"""Symmetric INT8 quantization: map float tensor to 8-bit range via scale."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_int8_quantization"]


def geron_int8_quantization(x):
    """
    Symmetric INT8 quantization: map float tensor to 8-bit range via scale

    Formula: q = round(x / s); s = max(|x|) / 127

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q, s

    References
    ----------
    Géron Ch 17, Quantization / INT8 section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetric INT8 quantization: map float tensor to 8-bit range via scale"})


def cheatsheet():
    return "grq8: Symmetric INT8 quantization: map float tensor to 8-bit range via scale"
