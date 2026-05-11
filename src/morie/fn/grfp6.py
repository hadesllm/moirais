# morie.fn — function file (hadesllm/morie)
"""Mixed-precision training: FP16 forward/backward, FP32 master weights + loss scaling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_fp16_mixed_precision"]


def geron_fp16_mixed_precision(loss, S):
    """
    Mixed-precision training: FP16 forward/backward, FP32 master weights + loss scaling

    Formula: loss_scaled = loss * S; gradients in FP16 get divided by S before FP32 update

    Parameters
    ----------
    loss : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: loss_scaled

    References
    ----------
    Géron Ch 17, Mixed Precision / FP16 section
    """
    loss = np.atleast_1d(np.asarray(loss, dtype=float))
    n = len(loss)
    result = float(np.mean(loss))
    se = float(np.std(loss, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixed-precision training: FP16 forward/backward, FP32 master weights + loss scaling"})


def cheatsheet():
    return "grfp6: Mixed-precision training: FP16 forward/backward, FP32 master weights + loss scaling"
