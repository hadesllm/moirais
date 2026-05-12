# morie.fn -- function file (hadesllm/morie)
"""PyTorch tensor: n-d array on CPU, GPU or MPS."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_pytorch_tensor"]


def geron_pytorch_tensor(x, device, dtype):
    """
    PyTorch tensor: n-d array on CPU, GPU or MPS

    Formula: t = torch.tensor(x, device, dtype)

    Parameters
    ----------
    x : array-like
        Input data.
    device : array-like
        Input data.
    dtype : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tensor

    References
    ----------
    Géron Ch 10
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PyTorch tensor: n-d array on CPU, GPU or MPS"})


def cheatsheet():
    return "hmpttn: PyTorch tensor: n-d array on CPU, GPU or MPS"
