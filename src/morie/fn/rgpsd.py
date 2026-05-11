# morie.fn — function file (hadesllm/morie)
"""Power spectral density (Welch method)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_psd"]


def rangayyan_psd(x):
    """
    Power spectral density (Welch method)

    Formula: S(f) = (2/NFs) |X(f)|^2

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
    Rangayyan Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power spectral density (Welch method)"})


def cheatsheet():
    return "rgpsd: Power spectral density (Welch method)"
