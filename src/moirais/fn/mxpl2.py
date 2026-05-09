# moirais.fn — function file (hadesllm/moirais)
r"""2D max pooling.

Downsampling via maximum operation over local windows.

References
----------
LeCun, Y., Bengio, Y., & Hinton, G. (2015).
Deep learning. Nature, 521(7553), 436-444.
"""

__all__ = ["mxpl2"]

import numpy as np
from ._richresult import RichResult


def mxpl2(x, pool_size=2, stride=None, padding='valid'):
    """
    2D max pooling.

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
    dict
        Keys: 'output', 'mask'.
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

    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=-np.inf)
    out_h = (x_padded.shape[1] - pool_size) // stride + 1
    out_w = (x_padded.shape[2] - pool_size) // stride + 1

    output = np.zeros((batch, out_h, out_w))
    mask = np.zeros_like(x_padded)

    for i in range(out_h):
        for j in range(out_w):
            window = x_padded[:, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[:, i, j] = np.max(window, axis=(1, 2))
            for b in range(batch):
                max_idx = np.unravel_index(
                    np.argmax(window[b]), window[b].shape
                )
                mask[b, i*stride+max_idx[0], j*stride+max_idx[1]] = 1

    return RichResult(payload={"output": output, "mask": mask})
