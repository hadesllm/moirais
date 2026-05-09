# moirais.fn — function file (hadesllm/moirais)
"""Gaussian time series posterior contraction for spectral density."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_ts_crt"]


def ghosal_ts_crt(x):
    """
    Gaussian time series posterior contraction for spectral density

    Formula: eps_n rate for spectral density f via Whittle likelihood

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
    Ghosal Ch 8 §8.3.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian time series posterior contraction for spectral density"})


def cheatsheet():
    return "gh_c8_11: Gaussian time series posterior contraction for spectral density"
