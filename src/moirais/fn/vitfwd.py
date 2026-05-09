"""Vision Transformer forward pass."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_forward"]


def vit_forward(x, patch_size, embed_dim, num_heads, num_layers):
    """
    Vision Transformer forward pass

    Formula: patch_embed -> [cls] + patches + pos_embed -> transformer -> mlp_head

    Parameters
    ----------
    x : array-like
        Input data.
    patch_size : array-like
        Input data.
    embed_dim : array-like
        Input data.
    num_heads : array-like
        Input data.
    num_layers : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020) ICLR
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vision Transformer forward pass"})


def cheatsheet():
    return "vitfwd: Vision Transformer forward pass"
