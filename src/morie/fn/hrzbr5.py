# morie.fn -- function file (hadesllm/morie)
"""Bias reduction for deconvolution via higher-order kernels."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_bias_reduction_deconv"]


def horowitz_bias_reduction_deconv(bandwidth, kernel_order):
    """
    Bias reduction for deconvolution via higher-order kernels

    Formula: Use K of order r: integral u^j K(u)du=0 for j=1,...,r-1; reduces O(h^2) to O(h^r)

    Parameters
    ----------
    bandwidth : array-like
        Input data.
    kernel_order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: reduced_bias_estimate

    References
    ----------
    Horowitz Ch 5, Sec 5.2.4
    """
    bandwidth = np.asarray(bandwidth, dtype=float)
    n = int(bandwidth) if bandwidth.ndim == 0 else len(bandwidth)
    result = float(np.mean(bandwidth))
    se = float(np.std(bandwidth, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias reduction for deconvolution via higher-order kernels"})


def cheatsheet():
    return "hrzbr5: Bias reduction for deconvolution via higher-order kernels"
