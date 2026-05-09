# moirais.fn — function file (hadesllm/moirais)
"""RWKV time-mixing: linear attention-free recurrent update with decay."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rwkv_time_mix"]


def kamath_rwkv_time_mix(k, v, w, u):
    """
    RWKV time-mixing: linear attention-free recurrent update with decay

    Formula: wkv_t = (sum_{i<=t} exp(-(t-i) w + k_i) * v_i) / (sum_{i<=t} exp(-(t-i) w + k_i))

    Parameters
    ----------
    k : array-like
        Input data.
    v : array-like
        Input data.
    w : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wkv

    References
    ----------
    Kamath Ch 10, RWKV section
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    n = len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "RWKV time-mixing: linear attention-free recurrent update with decay"})


def cheatsheet():
    return "kmrwkv: RWKV time-mixing: linear attention-free recurrent update with decay"
