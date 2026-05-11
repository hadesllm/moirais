# morie.fn — function file (hadesllm/morie)
r"""GELU activation (exact and approximation).

Gaussian Error Linear Unit.

References
----------
Hendrycks, D., & Gimpel, K. (2016).
Gaussian error linear units (gelus).
arXiv preprint arXiv:1606.08415.
"""

__all__ = ["gelua"]

import numpy as np
from scipy import special


def gelua(x, approximate=False, derivative=False, cdf=None):
    """
    GELU activation.

    Parameters
    ----------
    x : ndarray
        Input.
    approximate : bool, optional
        Use approximation. Default False (exact).
    derivative : bool, optional
        Return gradient. Default False.

    Returns
    -------
    ndarray
        Output or gradient.
    """
    x = np.asarray(x, dtype=float)

    if approximate:
        cdf = 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        ))
        if derivative:
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            return cdf + x * pdf
        else:
            return x * cdf
    else:
        cdf = 0.5 * (1.0 + special.erf(x / np.sqrt(2)))
        if derivative:
            pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
            return cdf + x * pdf
        else:
            return x * cdf
