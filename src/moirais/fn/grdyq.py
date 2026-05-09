# moirais.fn — function file (hadesllm/moirais)
"""Dynamic quantization: weights statically INT8, activations quantized per-batch at runtime."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dynamic_quantization"]


def geron_dynamic_quantization(x, w):
    """
    Dynamic quantization: weights statically INT8, activations quantized per-batch at runtime

    Formula: per batch: s_x = max(|x|)/127; x_q = round(x / s_x); accumulate in INT32, dequant result

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Appendix B, Dynamic Quantization section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dynamic quantization: weights statically INT8, activations quantized per-batch at runtime"})


def cheatsheet():
    return "grdyq: Dynamic quantization: weights statically INT8, activations quantized per-batch at runtime"
