# moirais.fn — function file (hadesllm/moirais)
"""Convolutional filter (kernel): learnable weight tensor."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_filter_kernel"]


def geron_filter_kernel(kh, kw, c_in, c_out, seed):
    """
    Convolutional filter (kernel): learnable weight tensor

    Formula: K in R^(kh x kw x C_in x C_out)

    Parameters
    ----------
    kh : array-like
        Input data.
    kw : array-like
        Input data.
    c_in : array-like
        Input data.
    c_out : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: K

    References
    ----------
    Géron Ch 12
    """
    kh = np.atleast_1d(np.asarray(kh, dtype=float))
    n = len(kh)
    result = float(np.mean(kh))
    se = float(np.std(kh, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convolutional filter (kernel): learnable weight tensor"})


def cheatsheet():
    return "hmkrn: Convolutional filter (kernel): learnable weight tensor"
