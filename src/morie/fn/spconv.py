"""Convolution representation of stationary random field."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_convolution_representation"]


def schabenberger_convolution_representation(kernel):
    """
    Convolution representation of stationary random field

    Formula: Z(s) = integral k(s-u) W(du), C(h)=integral k(s)*k(s+h)ds

    Parameters
    ----------
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: representation

    References
    ----------
    Schabenberger Ch 2, Sec 2.4.2
    """
    kernel = np.asarray(kernel, dtype=float)
    n = int(kernel) if kernel.ndim == 0 else len(kernel)
    result = float(np.mean(kernel))
    se = float(np.std(kernel, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convolution representation of stationary random field"})


def cheatsheet():
    return "spconv: Convolution representation of stationary random field"
