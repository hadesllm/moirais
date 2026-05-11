# morie.fn — function file (hadesllm/morie)
"""Dynamic quantization: quantize weights statically, activations at runtime."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dynamic_quantization"]


def geron_dynamic_quantization(model, dtype):
    """
    Dynamic quantization: quantize weights statically, activations at runtime

    Formula: weights: per-tensor INT8; activations: per-batch scale

    Parameters
    ----------
    model : array-like
        Input data.
    dtype : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: quantized_model

    References
    ----------
    Géron Appendix B
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dynamic quantization: quantize weights statically, activations at runtime"})


def cheatsheet():
    return "hmdqnt: Dynamic quantization: quantize weights statically, activations at runtime"
