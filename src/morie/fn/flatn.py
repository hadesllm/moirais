# morie.fn — function file (hadesllm/morie)
r"""Flatten layer.

Reshapes multi-dimensional input to 1D (per batch).

References
----------
Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
ImageNet classification with deep convolutional neural networks.
In NIPS (pp. 1097-1105).
"""

__all__ = ["flatn"]

import numpy as np


def flatn(x, batch_size=None):
    """
    Flatten layer.

    Parameters
    ----------
    x : ndarray
        Input, shape (batch, *dims).
    batch_size : int, optional
        Batch dimension size. If None, assumed first dimension.

    Returns
    -------
    ndarray
        Flattened, shape (batch, -1).
    """
    x = np.asarray(x, dtype=float)

    batch = x.shape[0]
    return x.reshape(batch, -1)
