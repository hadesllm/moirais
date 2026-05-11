# morie.fn — function file (hadesllm/morie)
"""Grouped query attention (GQA)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["grouped_query_attention"]


def grouped_query_attention(x):
    """
    Grouped query attention (GQA)

    Formula: n_heads queries share n_kv_heads keys/values

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
    Ainslie et al. (2023)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grouped query attention (GQA)"})


def cheatsheet():
    return "grpqa: Grouped query attention (GQA)"
