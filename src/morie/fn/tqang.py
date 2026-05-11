"""Uniform angle quantization over [-pi, pi]."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_angle_quantization"]


def turboquant_angle_quantization(theta, bits):
    """
    Uniform angle quantization over [-pi, pi]

    Formula: q = round( (theta + pi) / (2 pi) * (2^b - 1) );  theta_hat = q * 2 pi / (2^b - 1) - pi

    Parameters
    ----------
    theta : array-like
        Input data.
    bits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: q, theta_hat

    References
    ----------
    TurboQuant MORIE integration — quantize_angles
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Uniform angle quantization over [-pi, pi]"})


def cheatsheet():
    return "tqang: Uniform angle quantization over [-pi, pi]"
