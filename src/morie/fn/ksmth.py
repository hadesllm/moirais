# morie.fn — function file (hadesllm/morie)
"""Nadaraya-Watson kernel regression. 'We are what we repeatedly do. Excellence is not an act, but a habit. — Aristotle'"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def kernel_smooth(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: float | None = None,
) -> DescriptiveResult:
    r"""
    Nadaraya-Watson kernel regression with Gaussian kernel.

    .. math::

        \\hat{m}(x) = \\frac{\\sum_{i=1}^{n} K_h(x - x_i) y_i}
                           {\\sum_{i=1}^{n} K_h(x - x_i)}

    where :math:`K_h` is a Gaussian kernel with bandwidth *h*.

    :param x: Predictor array (1-D).
    :param y: Response array (1-D), same length as *x*.
    :param bandwidth: Kernel bandwidth. If None, uses Silverman's rule.
    :return: DescriptiveResult with fitted values in extra.
    :raises ValueError: If x and y differ in length or are too short.

    References
    ----------
    Nadaraya, E. A. (1964). On estimating regression. Theory of Probability
    and its Applications, 9(1), 141--142.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        raise ValueError("x and y must be 1-D arrays of equal length >= 3.")

    if bandwidth is None:
        bandwidth = 1.06 * np.std(x, ddof=1) * (len(x) ** (-1.0 / 5.0))
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be > 0, got {bandwidth}.")

    order = np.argsort(x)
    xs, ys = x[order], y[order]

    fitted = np.empty_like(xs)
    for i, xi in enumerate(xs):
        w = np.exp(-0.5 * ((xs - xi) / bandwidth) ** 2)
        fitted[i] = np.sum(w * ys) / np.sum(w)

    residual = ys - fitted
    rmse = float(np.sqrt(np.mean(residual**2)))

    return DescriptiveResult(
        name="Kernel Smooth (Nadaraya-Watson)",
        value=rmse,
        extra={
            "x_sorted": xs,
            "fitted": fitted,
            "residuals": residual,
            "rmse": rmse,
            "bandwidth": float(bandwidth),
            "n": len(x),
        },
    )


ksmth = kernel_smooth


def cheatsheet() -> str:
    return "kernel_smooth({}) -> Nadaraya-Watson kernel regression. 'The dark side clouds eve"
