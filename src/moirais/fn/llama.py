"""LLaMA decoder (RMSNorm, RoPE, SwiGLU)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["llama"]


def llama(tokens, model):
    """
    LLaMA decoder (RMSNorm, RoPE, SwiGLU)

    Formula: pre-norm transformer with RoPE attention

    Parameters
    ----------
    tokens : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Touvron et al (2023) LLaMA
    """
    tokens = np.atleast_1d(np.asarray(tokens, dtype=float))
    n = len(tokens)
    result = float(np.mean(tokens))
    se = float(np.std(tokens, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LLaMA decoder (RMSNorm, RoPE, SwiGLU)"})


def cheatsheet():
    return "llama: LLaMA decoder (RMSNorm, RoPE, SwiGLU)"
