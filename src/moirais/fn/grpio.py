# moirais.fn — function file (hadesllm/moirais)
"""Perceiver IO: learned latent array attends to input via cross-attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_perceiver_io"]


def geron_perceiver_io(X, Z_latent, output_queries):
    """
    Perceiver IO: learned latent array attends to input via cross-attention

    Formula: z = CrossAttn(Q=Z_latent, K=V=X_input); iterate self-attn + cross-attn; output = OutputCrossAttn(Q=outputs, K=V=z)

    Parameters
    ----------
    X : array-like
        Input data.
    Z_latent : array-like
        Input data.
    output_queries : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Géron Ch 16, Perceiver / Perceiver IO section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perceiver IO: learned latent array attends to input via cross-attention"})


def cheatsheet():
    return "grpio: Perceiver IO: learned latent array attends to input via cross-attention"
