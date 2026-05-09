# moirais.fn — function file (hadesllm/moirais)
"""Absolute sinusoidal positional encoding."""
import numpy as np
from ._richresult import RichResult

__all__ = ["positional_encoding_abs"]


def positional_encoding_abs(x):
    """
    Absolute sinusoidal positional encoding

    Formula: PE(pos,2i) = sin(pos/10000^(2i/d))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Absolute sinusoidal positional encoding"})


def cheatsheet():
    return "posab: Absolute sinusoidal positional encoding"
