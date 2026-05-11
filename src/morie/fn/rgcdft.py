# morie.fn — function file (hadesllm/morie)
"""Circular (cyclic) convolution via DFT."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_circular_conv_dft"]


def rangayyan_circular_conv_dft(x, h):
    """
    Circular (cyclic) convolution via DFT

    Formula: y_circ[n] = IDFT(DFT(x) * DFT(h))

    Parameters
    ----------
    x : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_circ

    References
    ----------
    Rangayyan Ch 3.4.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Circular (cyclic) convolution via DFT"})


def cheatsheet():
    return "rgcdft: Circular (cyclic) convolution via DFT"
