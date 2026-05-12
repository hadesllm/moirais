# morie.fn -- function file (hadesllm/morie)
"""Cosine kernel function."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["cskrn"]


def cskrn(u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Cosine kernel function.

    .. math::

        K(u) = \frac{\pi}{4}\cos\!\left(\frac{\pi u}{2}\right)
        \mathbf{1}(|u| \le 1)

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
    result = np.where(
        np.abs(u) <= 1.0,
        (np.pi / 4.0) * np.cos(np.pi * u / 2.0),
        0.0,
    )
    return float(result) if result.ndim == 0 else result


def cheatsheet() -> str:
    return "cskrn(u) -> Cosine kernel K(u) = pi/4*cos(pi*u/2)*I(|u|<=1)."
