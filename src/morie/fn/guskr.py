# morie.fn — function file (hadesllm/morie)
"""Gaussian kernel function."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["guskr"]


def guskr(u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Gaussian (normal) kernel function.

    .. math::

        K(u) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\tfrac{u^2}{2}\right)

    Parameters
    ----------
    u : float or np.ndarray
        Evaluation point(s).

    Returns
    -------
    float or np.ndarray
        Kernel value(s).

    References
    ----------
    Silverman, B. W. (1986). *Density Estimation for Statistics and Data
        Analysis*. Chapman & Hall. Table 3.1.
    """
    u = np.asarray(u, dtype=float)
    result = np.exp(-0.5 * u ** 2) / np.sqrt(2.0 * np.pi)
    return float(result) if result.ndim == 0 else result


def cheatsheet() -> str:
    return "guskr(u) -> Gaussian kernel K(u) = exp(-u^2/2)/sqrt(2*pi)."
