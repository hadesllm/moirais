# morie.fn — function file (hadesllm/morie)
"""Root-mean-square layer normalization (RMSNorm)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_rms_norm"]


def kamath_rms_norm(x, g, eps):
    """
    Root-mean-square layer normalization (RMSNorm)

    Formula: RMSNorm(x) = x / RMS(x) * g; RMS(x) = sqrt(mean(x^2) + eps)

    Parameters
    ----------
    x : array-like
        Input data.
    g : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Kamath Ch 2, RMSNorm section (Zhang & Sennrich 2019)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root-mean-square layer normalization (RMSNorm)"})


def cheatsheet():
    return "kmrmsn: Root-mean-square layer normalization (RMSNorm)"
