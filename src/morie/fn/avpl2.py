# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
r"""2D average pooling.

Downsampling via averaging over local windows.

References
----------
LeCun, Y., Bengio, Y., & Hinton, G. (2015).
Deep learning. Nature, 521(7553), 436-444.
"""

__all__ = ["avpl2"]

import numpy as np


def avpl2(x, pool_size=2, stride=None, padding='valid'):
    """
    2D average pooling.

    Parameters
    ----------
    x : ndarray
        Input, shape (batch, height, width) or (height, width).
    pool_size : int, optional
        Pool window size. Default 2.
    stride : int, optional
        Stride. Default pool_size.
    padding : str, optional
        'valid' or 'same'. Default 'valid'.

    Returns
    -------
    ndarray
        Pooled output.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 2:
        x = x[np.newaxis, :, :]

    batch, height, width = x.shape
    stride = stride or pool_size

    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (pool_size - 1) // 2
        pad_w = (pool_size - 1) // 2
    else:
        raise ValueError("padding must be 'valid' or 'same'")

    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    out_h = (x_padded.shape[1] - pool_size) // stride + 1
    out_w = (x_padded.shape[2] - pool_size) // stride + 1

    output = np.zeros((batch, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[:, i, j] = np.mean(
                x_padded[:, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size],
                axis=(1, 2)
            )

    return output
