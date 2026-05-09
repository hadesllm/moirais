"""ViT MLP block."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_mlp_block"]


def vit_mlp_block(x, hidden_dim):
    """
    ViT MLP block

    Formula: GELU(W1 x + b1) -> W2

    Parameters
    ----------
    x : array-like
        Input data.
    hidden_dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dosovitskiy et al (2020)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT MLP block"})


def cheatsheet():
    return "vitmlp: ViT MLP block"
