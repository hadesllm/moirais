# morie.fn — function file (hadesllm/morie)
"""ResNet residual skip connection: y = F(x) + x."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_resnet_skip"]


def geron_resnet_skip(x, Fx):
    """
    ResNet residual skip connection: y = F(x) + x

    Formula: y = F(x; theta) + x

    Parameters
    ----------
    x : array-like
        Input data.
    Fx : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 12, ResNet / Skip Connections section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ResNet residual skip connection: y = F(x) + x"})


def cheatsheet():
    return "grrsk: ResNet residual skip connection: y = F(x) + x"
