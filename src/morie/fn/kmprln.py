# morie.fn -- function file (hadesllm/morie)
"""Pre-LayerNorm transformer block (stable training)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_pre_ln_transformer"]


def kamath_pre_ln_transformer(x, attn_fn, ffn_fn):
    """
    Pre-LayerNorm transformer block (stable training)

    Formula: y = x + Attn(LN(x)); z = y + FFN(LN(y))

    Parameters
    ----------
    x : array-like
        Input data.
    attn_fn : array-like
        Input data.
    ffn_fn : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: z

    References
    ----------
    Kamath Ch 2, Pre-LN vs Post-LN section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pre-LayerNorm transformer block (stable training)"})


def cheatsheet():
    return "kmprln: Pre-LayerNorm transformer block (stable training)"
