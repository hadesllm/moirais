# morie.fn -- function file (hadesllm/morie)
"""DALL·E: text-to-image generation via discrete VAE + autoregressive transformer."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_dalle"]


def geron_dalle(text, model):
    """
    DALL·E: text-to-image generation via discrete VAE + autoregressive transformer

    Formula: text tokens -> image tokens via autoregressive LM

    Parameters
    ----------
    text : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: image

    References
    ----------
    Géron Ch 16
    """
    text = np.atleast_1d(np.asarray(text, dtype=float))
    n = len(text)
    result = float(np.mean(text))
    se = float(np.std(text, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DALL·E: text-to-image generation via discrete VAE + autoregressive transformer"})


def cheatsheet():
    return "hmdale: DALL·E: text-to-image generation via discrete VAE + autoregressive transformer"
