# moirais.fn — function file (hadesllm/moirais)
"""Gradient clipping by global norm."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gradient_clipping"]


def gradient_clipping(x):
    """
    Gradient clipping by global norm

    Formula: g = g * max_norm / max(||g||, max_norm)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pascanu et al. (2013)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient clipping by global norm"})


def cheatsheet():
    return "grdcl: Gradient clipping by global norm"
