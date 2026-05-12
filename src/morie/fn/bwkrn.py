# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Biweight (quartic) kernel function."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["bwkrn"]


def bwkrn(u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Biweight (quartic) kernel function.

    .. math::

        K(u) = \frac{15}{16}(1 - u^2)^2 \mathbf{1}(|u| \le 1)

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
    result = np.where(np.abs(u) <= 1.0, (15.0 / 16.0) * (1.0 - u ** 2) ** 2, 0.0)
    return float(result) if result.ndim == 0 else result


def cheatsheet() -> str:
    return "bwkrn(u) -> Biweight kernel K(u) = 15/16*(1-u^2)^2*I(|u|<=1)."
