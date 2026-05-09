"""Triweight kernel function."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["trkrn"]


def trkrn(u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Triweight kernel function.

    .. math::

        K(u) = \frac{35}{32}(1 - u^2)^3 \mathbf{1}(|u| \le 1)

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
    Wand, M. P. & Jones, M. C. (1995). *Kernel Smoothing*. Chapman & Hall.
        Table 2.1.
    """
    u = np.asarray(u, dtype=float)
    result = np.where(np.abs(u) <= 1.0, (35.0 / 32.0) * (1.0 - u ** 2) ** 3, 0.0)
    return float(result) if result.ndim == 0 else result


def cheatsheet() -> str:
    return "trkrn(u) -> Triweight kernel K(u) = 35/32*(1-u^2)^3*I(|u|<=1)."
