# morie.fn -- function file (hadesllm/morie)
r"""RMS normalization.

Normalizes by root-mean-square of activations.

References
----------
Zhang, B., & Sennrich, R. (2019).
Root mean square layer normalization.
In NeurIPS.
"""

__all__ = ["rmsno"]

import numpy as np
from ._richresult import RichResult


def rmsno(
    x,
    epsilon=1e-6,
    gamma=None,
):
    """
    RMS normalization.

    Parameters
    ----------
    x : ndarray
        Input, shape (batch, ..., features).
    epsilon : float, optional
        Numerical stability. Default 1e-6.
    gamma : ndarray, optional
        Scale, shape (features,).

    Returns
    -------
    dict
        Keys: 'output', 'rms'.
    """
    x = np.asarray(x, dtype=float)

    axis = tuple(range(1, x.ndim))
    rms = np.sqrt(np.mean(x**2, axis=axis, keepdims=True) + epsilon)

    x_norm = x / rms

    if gamma is not None:
        x_norm = x_norm * gamma

    return RichResult(payload={"output": x_norm, "rms": rms})
