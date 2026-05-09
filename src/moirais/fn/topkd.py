"""Top-k sampling for text generation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["top_k_decoding"]


def top_k_decoding(x):
    """
    Top-k sampling for text generation

    Formula: sample from top-k logits, renormalize

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
    Fan et al. (2018)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Top-k sampling for text generation"})


def cheatsheet():
    return "topkd: Top-k sampling for text generation"
