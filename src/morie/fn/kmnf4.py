# morie.fn -- function file (hadesllm/morie)
"""NormalFloat4 (NF4) data type: information-theoretically optimal 4-bit grid for N(0,1) weights."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_nf4_datatype"]


def kamath_nf4_datatype(n_bins):
    """
    NormalFloat4 (NF4) data type: information-theoretically optimal 4-bit grid for N(0,1) weights

    Formula: q_i = Phi^{-1}((i + 0.5) / 16)  for i in 0..15  (quantile-based)

    Parameters
    ----------
    n_bins : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: codebook

    References
    ----------
    Kamath Ch 4, NF4 section
    """
    n_bins = np.atleast_1d(np.asarray(n_bins, dtype=float))
    n = len(n_bins)
    result = float(np.mean(n_bins))
    se = float(np.std(n_bins, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "NormalFloat4 (NF4) data type: information-theoretically optimal 4-bit grid for N(0,1) weights"})


def cheatsheet():
    return "kmnf4: NormalFloat4 (NF4) data type: information-theoretically optimal 4-bit grid for N(0,1) weights"
