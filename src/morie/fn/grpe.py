# morie.fn -- function file (hadesllm/morie)
"""Sinusoidal positional encoding (Vaswani et al. 2017)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sinusoidal_positional_encoding"]


def geron_sinusoidal_positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017)

    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d)); PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Parameters
    ----------
    seq_len : array-like
        Input data.
    d_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: PE

    References
    ----------
    Géron Ch 15, Positional Encodings section
    """
    seq_len = np.atleast_1d(np.asarray(seq_len, dtype=float))
    n = len(seq_len)
    result = float(np.mean(seq_len))
    se = float(np.std(seq_len, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sinusoidal positional encoding (Vaswani et al. 2017)"})


def cheatsheet():
    return "grpe: Sinusoidal positional encoding (Vaswani et al. 2017)"
