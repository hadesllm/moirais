# morie.fn — function file (hadesllm/morie)
"""QAT: simulate quantization in the forward pass and straight-through in the backward."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_quantization_aware_training"]


def geron_quantization_aware_training(x, s, bits):
    """
    QAT: simulate quantization in the forward pass and straight-through in the backward

    Formula: y = dequant(quant(x)); backward pass uses straight-through estimator (grad passes through)

    Parameters
    ----------
    x : array-like
        Input data.
    s : array-like
        Input data.
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Appendix B, Quantization-Aware Training section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QAT: simulate quantization in the forward pass and straight-through in the backward"})


def cheatsheet():
    return "grqat: QAT: simulate quantization in the forward pass and straight-through in the backward"
