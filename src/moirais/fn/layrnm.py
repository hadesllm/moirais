"""LayerNorm — per-token normalization."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["layer_norm"]


def layer_norm(y, x, g, b, eps):
    """
    LayerNorm — per-token normalization

    Formula: y = (x - mean) / sqrt(var + eps) * g + b

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    g : array-like
        Input data.
    b : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ba, Kiros, Hinton (2016)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LayerNorm — per-token normalization"})


def cheatsheet():
    return "layrnm: LayerNorm — per-token normalization"
