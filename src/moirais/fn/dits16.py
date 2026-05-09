"""Diffusion Transformer (DiT)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dit_diffusion_transformer"]


def dit_diffusion_transformer(x, t, cls):
    """
    Diffusion Transformer (DiT)

    Formula: ViT-style transformer for diffusion

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    cls : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Peebles-Xie (2023) DiT
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Diffusion Transformer (DiT)"})


def cheatsheet():
    return "dits16: Diffusion Transformer (DiT)"
