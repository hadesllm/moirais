r"""Swish activation function (SiLU).

Smooth, non-monotonic activation.

References
----------
Ramachandran, P., Zoph, B., & Le, Q. V. (2017).
Searching for activation functions.
arXiv preprint arXiv:1710.05941.
"""

__all__ = ["swish"]

import numpy as np


def swish(x, derivative=False):
    """
    Swish (SiLU) activation.

    Parameters
    ----------
    x : ndarray
        Input.
    derivative : bool, optional
        Return gradient. Default False.

    Returns
    -------
    ndarray
        Output or gradient.
    """
    x = np.asarray(x, dtype=float)
    sigmoid = 1.0 / (1.0 + np.exp(-x))

    if derivative:
        return sigmoid * (1.0 + x * (1.0 - sigmoid))
    else:
        return x * sigmoid
