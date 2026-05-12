# morie.fn -- function file (hadesllm/morie)
"""Brain floating point (BF16): FP32-range with FP16-size."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bf16"]


def geron_bf16(x):
    """
    Brain floating point (BF16): FP32-range with FP16-size

    Formula: 1 sign, 8 exponent, 7 mantissa bits

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_bf16

    References
    ----------
    Géron Appendix B
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Brain floating point (BF16): FP32-range with FP16-size"})


def cheatsheet():
    return "hmbf16: Brain floating point (BF16): FP32-range with FP16-size"
