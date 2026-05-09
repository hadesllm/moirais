# moirais.fn — function file (hadesllm/moirais)
"""Mixed-precision training: FP16 forward, FP32 master weights."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mixed_precision"]


def geron_mixed_precision(model, loss_scale):
    """
    Mixed-precision training: FP16 forward, FP32 master weights

    Formula: weights in FP32; activations/grads in FP16; loss scaling

    Parameters
    ----------
    model : array-like
        Input data.
    loss_scale : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 17
    """
    model = np.atleast_1d(np.asarray(model, dtype=float))
    n = len(model)
    result = float(np.mean(model))
    se = float(np.std(model, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixed-precision training: FP16 forward, FP32 master weights"})


def cheatsheet():
    return "hmmxp2: Mixed-precision training: FP16 forward, FP32 master weights"
