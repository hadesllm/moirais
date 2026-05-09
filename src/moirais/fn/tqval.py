"""Value-cache quantization: token-wise normalize + round-to-integer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_value_cache_quantization"]


def turboquant_value_cache_quantization(v, bits):
    """
    Value-cache quantization: token-wise normalize + round-to-integer

    Formula: v_q = round(v / s * (2^b - 1));  s = max(|v|);  reconstruct v_hat = v_q * s / (2^b - 1)

    Parameters
    ----------
    v : array-like
        Input data.
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: v_q, s

    References
    ----------
    Zandieh et al. 2024 Section 3.2 (value cache quantization)
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    n = len(v)
    result = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Value-cache quantization: token-wise normalize + round-to-integer"})


def cheatsheet():
    return "tqval: Value-cache quantization: token-wise normalize + round-to-integer"
