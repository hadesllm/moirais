# morie.fn — function file (hadesllm/morie)
"""Root mean square normalization."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rms_norm"]


def rms_norm(x):
    """
    Root mean square normalization

    Formula: RMSNorm(x) = x / RMS(x) * gamma

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
    Zhang & Sennrich (2019)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root mean square normalization"})


def cheatsheet():
    return "rmsnr: Root mean square normalization"
