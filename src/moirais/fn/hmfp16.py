# moirais.fn — function file (hadesllm/moirais)
"""FP16 half precision."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_fp16_quant"]


def geron_fp16_quant(x):
    """
    FP16 half precision

    Formula: float16 representation (1 sign, 5 exp, 10 mantissa)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x_fp16

    References
    ----------
    Géron Ch 17
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "FP16 half precision"})


def cheatsheet():
    return "hmfp16: FP16 half precision"
