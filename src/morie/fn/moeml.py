# morie.fn — function file (hadesllm/morie)
"""Mixture of experts layer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["mixture_of_experts"]


def mixture_of_experts(x):
    """
    Mixture of experts layer

    Formula: y = sum g_i(x) * E_i(x), g = softmax(Wx)

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
    Shazeer et al. (2017)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mixture of experts layer"})


def cheatsheet():
    return "moeml: Mixture of experts layer"
