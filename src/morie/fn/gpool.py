# morie.fn -- function file (hadesllm/morie)
r"""Global average pooling.

Reduces spatial dimensions to 1x1.

References
----------
Lin, M., Chen, Q., & Yan, S. (2013).
Network in network.
arXiv preprint arXiv:1312.4400.
"""

__all__ = ["gpool"]

import numpy as np


def gpool(x):
    """
    Global average pooling.

    Parameters
    ----------
    x : ndarray
        Input, shape (batch, height, width) or (batch, channels, height, width).

    Returns
    -------
    ndarray
        Spatially averaged output.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim < 2:
        raise ValueError("Input must be at least 2D")

    if x.ndim == 3:
        spatial_axes = (1, 2)
    else:
        spatial_axes = tuple(range(2, x.ndim))

    output = np.mean(x, axis=spatial_axes)

    return output
