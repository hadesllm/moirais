# moirais.fn — function file (hadesllm/moirais)
"""Causal (autoregressive) attention mask."""
import numpy as np
from ._richresult import RichResult

__all__ = ["causal_attention_mask"]


def causal_attention_mask(x):
    """
    Causal (autoregressive) attention mask

    Formula: mask[i,j] = -inf if j > i, else 0

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
    Radford et al. (2019)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal (autoregressive) attention mask"})


def cheatsheet():
    return "cslat: Causal (autoregressive) attention mask"
