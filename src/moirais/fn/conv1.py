# moirais.fn — function file (hadesllm/moirais)
r"""1D convolution (valid/same/full).

Applies sliding window operation on 1D signal.

References
----------
LeCun, Y., Bengio, Y., & Hinton, G. (2015).
Deep learning. Nature, 521(7553), 436-444.
"""

__all__ = ["conv1"]

import numpy as np


def conv1(x, kernel, padding='valid', stride=1):
    """
    1D convolution.

    Parameters
    ----------
    x : ndarray
        Input signal, shape (length,) or (batch, length).
    kernel : ndarray
        Convolution kernel, shape (kernel_size,).
    padding : str, optional
        'valid', 'same', or 'full'. Default 'valid'.
    stride : int, optional
        Step size. Default 1.

    Returns
    -------
    ndarray
        Convolved output.
    """
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if x.ndim == 1:
        x = x[np.newaxis, :]
    batch, length = x.shape
    k_size = kernel.shape[0]

    if padding == 'valid':
        pad = 0
    elif padding == 'same':
        pad = (k_size - 1) // 2
    elif padding == 'full':
        pad = k_size - 1
    else:
        raise ValueError("padding must be 'valid', 'same', or 'full'")

    x_padded = np.pad(x, ((0, 0), (pad, pad)), mode='constant')
    out_length = (x_padded.shape[1] - k_size) // stride + 1

    output = np.zeros((batch, out_length))
    for i in range(out_length):
        output[:, i] = np.sum(x_padded[:, i*stride:i*stride+k_size] * kernel, axis=1)

    return output
