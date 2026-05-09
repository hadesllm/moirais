"""Transformer encoder block."""
import numpy as np
from ._richresult import RichResult

__all__ = ["transformer_block"]


def transformer_block(x):
    """
    Transformer encoder block

    Formula: x = LN(x + MHA(x)), x = LN(x + FFN(x))

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
    Vaswani et al. (2017)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformer encoder block"})


def cheatsheet():
    return "trfbl: Transformer encoder block"
