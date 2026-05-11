# morie.fn — function file (hadesllm/morie)
r"""ReLU6 activation function.

Clamps ReLU output to [0, 6].

References
----------
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
MobileNetV2: Inverted residuals and linear bottlenecks.
In CVPR (pp. 4510-4520).
"""

__all__ = ["relu6"]

import numpy as np


def relu6(x, derivative=False):
    """
    ReLU6 activation.

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

    if derivative:
        return np.where((x >= 0) & (x <= 6), 1.0, 0.0)
    else:
        return np.clip(x, 0, 6)
