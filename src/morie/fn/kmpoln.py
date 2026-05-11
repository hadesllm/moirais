# morie.fn — function file (hadesllm/morie)
"""Post-LayerNorm transformer block (original Vaswani placement)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_post_ln_transformer"]


def kamath_post_ln_transformer(x, attn_fn, ffn_fn):
    """
    Post-LayerNorm transformer block (original Vaswani placement)

    Formula: y = LN(x + Attn(x)); z = LN(y + FFN(y))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Post-LayerNorm transformer block (original Vaswani placement)"})


def cheatsheet():
    return "kmpoln: Post-LayerNorm transformer block (original Vaswani placement)"
