# morie.fn — function file (hadesllm/morie)
"""Epanechnikov kernel function."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["epkrn"]


def epkrn(u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Epanechnikov kernel function.

    .. math::

        K(u) = \frac{3}{4}(1 - u^2) \mathbf{1}(|u| \le 1)

    This is the theoretically MISE-optimal kernel among all symmetric
    density functions.

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
    Epanechnikov, V. A. (1969). Non-parametric estimation of a
        multivariate probability density. *Theory of Probability and its
        Applications*, 14(1), 153-158.
    """
    u = np.asarray(u, dtype=float)
    result = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u ** 2), 0.0)
    return float(result) if result.ndim == 0 else result


def cheatsheet() -> str:
    return "epkrn(u) -> Epanechnikov kernel K(u) = 3/4*(1-u^2)*I(|u|<=1)."
