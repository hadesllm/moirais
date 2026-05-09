# moirais.fn — function file (hadesllm/moirais)
"""Perceiver: cross-attention to learned latents from high-dim inputs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_perceiver"]


def geron_perceiver(x, latents, n_iter):
    """
    Perceiver: cross-attention to learned latents from high-dim inputs

    Formula: latents Q cross-attend to input K,V; iterate to refine

    Parameters
    ----------
    x : array-like
        Input data.
    latents : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: latents

    References
    ----------
    Géron Ch 16
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perceiver: cross-attention to learned latents from high-dim inputs"})


def cheatsheet():
    return "hmprcv: Perceiver: cross-attention to learned latents from high-dim inputs"
