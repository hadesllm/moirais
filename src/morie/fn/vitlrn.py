"""ViT pre-LayerNorm."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vit_layer_norm"]


def vit_layer_norm(x, gamma, beta):
    """
    ViT pre-LayerNorm

    Formula: y = (x - mu)/sigma * gamma + beta

    Parameters
    ----------
    x : array-like
        Input data.
    gamma : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ba-Kiros-Hinton (2016)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ViT pre-LayerNorm"})


def cheatsheet():
    return "vitlrn: ViT pre-LayerNorm"
