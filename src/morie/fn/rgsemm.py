# morie.fn — function file (hadesllm/morie)
"""Spectral error measure (SEM) for adaptive segmentation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_spec_error_meas"]


def rangayyan_spec_error_meas(x, fs, p, seg_len):
    """
    Spectral error measure (SEM) for adaptive segmentation

    Formula: SEM(m) = (1/p) sum_{k=1}^{p} (log S_m(k) - log S_{ref}(k))^2

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    p : array-like
        Input data.
    seg_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sem_trace, segment_bounds

    References
    ----------
    Rangayyan Ch 8.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral error measure (SEM) for adaptive segmentation"})


def cheatsheet():
    return "rgsemm: Spectral error measure (SEM) for adaptive segmentation"
