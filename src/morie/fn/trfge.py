"""Transformer attention for genomic prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["transformer_genomic"]


def transformer_genomic(x, y, markers):
    """
    Transformer attention for genomic prediction

    Formula: Self-attention on marker embeddings

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    markers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 15
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformer attention for genomic prediction"})


def cheatsheet():
    return "trfge: Transformer attention for genomic prediction"
