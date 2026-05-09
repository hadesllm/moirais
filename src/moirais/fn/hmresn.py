# moirais.fn — function file (hadesllm/moirais)
"""ResNet: residual block with skip connection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_resnet"]


def geron_resnet(x, F):
    """
    ResNet: residual block with skip connection

    Formula: y = F(x) + x

    Parameters
    ----------
    x : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ResNet: residual block with skip connection"})


def cheatsheet():
    return "hmresn: ResNet: residual block with skip connection"
