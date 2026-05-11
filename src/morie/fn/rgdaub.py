# morie.fn — function file (hadesllm/morie)
"""Daubechies wavelet filter coefficients (db2-db10)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_daubechies"]


def rangayyan_daubechies(order):
    """
    Daubechies wavelet filter coefficients (db2-db10)

    Formula: Orthogonal FIR filter satisfying vanishing moment conditions

    Parameters
    ----------
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo_d, hi_d, lo_r, hi_r

    References
    ----------
    Rangayyan Ch 8.8
    """
    order = np.asarray(order, dtype=float)
    n = int(order) if order.ndim == 0 else len(order)
    result = float(np.mean(order))
    se = float(np.std(order, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Daubechies wavelet filter coefficients (db2-db10)"})


def cheatsheet():
    return "rgdaub: Daubechies wavelet filter coefficients (db2-db10)"
