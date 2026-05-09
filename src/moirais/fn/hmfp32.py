# moirais.fn — function file (hadesllm/moirais)
"""Single-precision (FP32) representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_fp32"]


def geron_fp32(x):
    """
    Single-precision (FP32) representation

    Formula: 1 sign, 8 exponent, 23 mantissa bits

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_fp32

    References
    ----------
    Géron Appendix B
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Single-precision (FP32) representation"})


def cheatsheet():
    return "hmfp32: Single-precision (FP32) representation"
