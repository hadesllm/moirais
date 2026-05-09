# moirais.fn — function file (hadesllm/moirais)
"""Double quantization: quantize the (FP32) quantization constants themselves."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_double_quantization"]


def kamath_double_quantization(scales_fp32):
    """
    Double quantization: quantize the (FP32) quantization constants themselves

    Formula: quantize per-block scales to 8-bit with a shared FP32 constant

    Parameters
    ----------
    scales_fp32 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scales_int8, shared_const

    References
    ----------
    Kamath Ch 4, Double Quantization section
    """
    scales_fp32 = np.atleast_1d(np.asarray(scales_fp32, dtype=float))
    n = len(scales_fp32)
    result = float(np.mean(scales_fp32))
    se = float(np.std(scales_fp32, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double quantization: quantize the (FP32) quantization constants themselves"})


def cheatsheet():
    return "kmdbq: Double quantization: quantize the (FP32) quantization constants themselves"
