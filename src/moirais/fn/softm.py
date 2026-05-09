r"""Softmax activation (numerically stable).

Normalizes output to probability distribution.

References
----------
Goodfellow, I., Bengio, Y., & Courville, A. (2016).
Deep Learning. MIT press.
"""

__all__ = ["softm"]

import numpy as np
from ._richresult import RichResult


def softm(x, axis=-1, derivative=False):
    """
    Softmax activation (numerically stable).

    Parameters
    ----------
    x : ndarray
        Logits, shape (..., n_classes).
    axis : int, optional
        Axis along which to normalize. Default -1.
    derivative : bool, optional
        Return Jacobian. Default False.

    Returns
    -------
    ndarray or dict
        If derivative=False: softmax probabilities.
        If derivative=True: dict with 'output' and 'jacobian'.
    """
    x = np.asarray(x, dtype=float)

    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    softmax = exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    if derivative:
        n = softmax.shape[axis]
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                jacobian[i, j] = softmax[i] * (int(i == j) - softmax[j])
        return RichResult(payload={"output": softmax, "jacobian": jacobian})
    else:
        return softmax
