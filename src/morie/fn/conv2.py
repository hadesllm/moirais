# morie.fn — function file (hadesllm/morie)
r"""2D convolution (forward pass).

Applies sliding 2D kernel on image.

References
----------
LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., Hubbard, W., & Jackel, L. D. (1990).
Handwritten digit recognition with a back-propagation network.
In Advances in neural information processing systems (pp. 396-404).
"""

__all__ = ["conv2"]

import numpy as np


def conv2(x, kernel, padding='valid', stride=1):
    """
    2D convolution.

    Parameters
    ----------
    x : ndarray
        Input image, shape (height, width) or (batch, height, width).
    kernel : ndarray
        Kernel, shape (k_h, k_w).
    padding : str, optional
        'valid' or 'same'. Default 'valid'.
    stride : int, optional
        Step size. Default 1.

    Returns
    -------
    ndarray
        Convolved output.
    """
    x = np.asarray(x, dtype=float)
    kernel = np.asarray(kernel, dtype=float)

    if x.ndim == 2:
        x = x[np.newaxis, :, :]
    batch, height, width = x.shape
    k_h, k_w = kernel.shape

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (k_h - 1) // 2
        pad_w = (k_w - 1) // 2
    else:
        raise ValueError("padding must be 'valid' or 'same'")

    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    out_h = (x_padded.shape[1] - k_h) // stride + 1
    out_w = (x_padded.shape[2] - k_w) // stride + 1

    output = np.zeros((batch, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.sum(
                x_padded[:, i*stride:i*stride+k_h, j*stride:j*stride+k_w] * kernel,
                axis=(1, 2)
            )

    return output
