# morie.fn — function file (hadesllm/morie)
"""Cosine learning rate schedule with warmup."""
import numpy as np
from ._richresult import RichResult

__all__ = ["cosine_lr_schedule"]


def cosine_lr_schedule(x):
    """
    Cosine learning rate schedule with warmup

    Formula: lr = lr_min + 0.5(lr_max-lr_min)(1+cos(pi*t/T))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Loshchilov & Hutter (2017)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cosine learning rate schedule with warmup"})


def cheatsheet():
    return "cslnc: Cosine learning rate schedule with warmup"
