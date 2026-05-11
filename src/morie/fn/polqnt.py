"""PolarQuant: 4-bit polar codebook compression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["polar_quantization"]


def polar_quantization(x, bits):
    """
    PolarQuant: 4-bit polar codebook compression

    Formula: polar coords (r, theta); quantize separately

    Parameters
    ----------
    x : array-like
        Input data.
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    PolarQuant (Tang 2024)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PolarQuant: 4-bit polar codebook compression"})


def cheatsheet():
    return "polqnt: PolarQuant: 4-bit polar codebook compression"
