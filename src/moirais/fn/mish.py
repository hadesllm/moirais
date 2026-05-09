# moirais.fn — function file (hadesllm/moirais)
r"""Mish activation function.

Self-regularizing smooth activation.

References
----------
Misra, D. (2019).
Mish: A self regularized activation function.
arXiv preprint arXiv:1908.03682.
"""

__all__ = ["mish"]

import numpy as np


def mish(x, derivative=False):
    """
    Mish activation.

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
    tanh_arg = 1.0 / (1.0 + np.exp(-x))
    softplus = np.log(1.0 + np.exp(x))

    if derivative:
        sech2 = 1.0 - tanh_arg**2
        sigmoid = 1.0 / (1.0 + np.exp(-softplus))
        return sigmoid * (x * sech2 + tanh_arg)
    else:
        return x * np.tanh(softplus)
