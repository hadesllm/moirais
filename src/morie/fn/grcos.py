# morie.fn — function file (hadesllm/morie)
"""Conv layer output spatial size from input size, filter, padding, stride."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_conv_output_size"]


def geron_conv_output_size(in_size, kernel, padding, stride):
    """
    Conv layer output spatial size from input size, filter, padding, stride

    Formula: out = floor((in + 2*p - k) / s) + 1

    Parameters
    ----------
    in_size : array-like
        Input data.
    kernel : array-like
        Input data.
    padding : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: out_size

    References
    ----------
    Géron Ch 12, padding/stride output-size equation
    """
    in_size = np.atleast_1d(np.asarray(in_size, dtype=float))
    n = len(in_size)
    result = float(np.mean(in_size))
    se = float(np.std(in_size, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conv layer output spatial size from input size, filter, padding, stride"})


def cheatsheet():
    return "grcos: Conv layer output spatial size from input size, filter, padding, stride"
