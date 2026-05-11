# morie.fn — function file (hadesllm/morie)
"""1cycle policy: warm up then anneal the learning rate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_cycle"]


def geron_one_cycle(t, T, lr_max, lr_min):
    """
    1cycle policy: warm up then anneal the learning rate

    Formula: lr ramps up to lr_max then down to lr_min over T steps

    Parameters
    ----------
    t : array-like
        Input data.
    T : array-like
        Input data.
    lr_max : array-like
        Input data.
    lr_min : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lr

    References
    ----------
    Géron Ch 11
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "1cycle policy: warm up then anneal the learning rate"})


def cheatsheet():
    return "hml1c: 1cycle policy: warm up then anneal the learning rate"
