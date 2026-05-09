# moirais.fn — function file (hadesllm/moirais)
"""BF16 representation: 1 sign + 8 exponent + 7 mantissa bits (same exponent range as FP32)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bf16_range"]


def geron_bf16_range(x):
    """
    BF16 representation: 1 sign + 8 exponent + 7 mantissa bits (same exponent range as FP32)

    Formula: bf16 covers same dynamic range as fp32 (~1.18e-38..3.4e38) with 7-bit mantissa precision

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bf16

    References
    ----------
    Géron Appendix B, BF16 section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "BF16 representation: 1 sign + 8 exponent + 7 mantissa bits (same exponent range as FP32)"})


def cheatsheet():
    return "grbf16: BF16 representation: 1 sign + 8 exponent + 7 mantissa bits (same exponent range as FP32)"
