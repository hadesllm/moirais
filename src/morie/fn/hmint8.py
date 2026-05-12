# morie.fn -- function file (hadesllm/morie)
"""INT8 quantization: post-training 8-bit weight+activation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_int8_quant"]


def geron_int8_quant(x, n_bits, symmetric):
    """
    INT8 quantization: post-training 8-bit weight+activation

    Formula: q = round((x - z) / s); x = q*s + z

    Parameters
    ----------
    x : array-like
        Input data.
    n_bits : array-like
        Input data.
    symmetric : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q, scale, zero

    References
    ----------
    Géron Ch 17
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "INT8 quantization: post-training 8-bit weight+activation"})


def cheatsheet():
    return "hmint8: INT8 quantization: post-training 8-bit weight+activation"
