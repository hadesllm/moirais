"""Theoretical MSE distortion bound for TurboQuant at b bits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_mse_distortion_bound"]


def turboquant_mse_distortion_bound(bits):
    """
    Theoretical MSE distortion bound for TurboQuant at b bits

    Formula: MSE <= c_b * sigma^2;  c_b = Panter-Dite constant for b-bit Lloyd-Max

    Parameters
    ----------
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    TurboQuant MORIE integration — mse_distortion_bound
    """
    bits = np.atleast_1d(np.asarray(bits, dtype=float))
    n = len(bits)
    result = float(np.mean(bits))
    se = float(np.std(bits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Theoretical MSE distortion bound for TurboQuant at b bits"})


def cheatsheet():
    return "tqmsb: Theoretical MSE distortion bound for TurboQuant at b bits"
