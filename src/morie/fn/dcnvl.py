# morie.fn -- function file (hadesllm/morie)
r"""Deconvolution density estimation under measurement error.

Estimates the density :math:`f_X` of a latent variable :math:`X` from
observations :math:`W = X + U` where :math:`U` is measurement error
with known characteristic function (e.g., normal or Laplace).

Uses the Fourier deconvolution approach:

.. math::

    \\hat{f}_X(x) = \\frac{1}{2\\pi} \\int
        e^{-itx} \\,
        \\frac{\\hat{\\phi}_W(t)}{\\phi_U(t)} \\,
        K^*(th) \\, dt

References
----------
Fan, J. (1991). On the optimal rates of convergence for nonparametric
    deconvolution problems. *Annals of Statistics*, 19(3), 1257--1272.
Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods in
    Econometrics*. Springer. Chapter 8.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def dcnvl(
    W: np.ndarray,
    *,
    error_sd: float = 1.0,
    error_type: str = "normal",
    grid_points: int = 256,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    r"""Deconvolution density estimator.

    Parameters
    ----------
    W : np.ndarray
        Observed (contaminated) data, shape ``(n,)``.
    error_sd : float
        Standard deviation of the measurement error.
    error_type : str
        ``"normal"`` (supersmooth) or ``"laplace"`` (ordinary smooth).
    grid_points : int
        Number of evaluation points.
    bandwidth : float or None
        Fourier-space bandwidth; if *None*, uses a plug-in rule.

    Returns
    -------
    dict[str, Any]
        ``x_grid``, ``density`` (estimated :math:`f_X`), ``bandwidth``,
        ``n``, ``method``.
    """
    W = np.asarray(W, dtype=float).ravel()
    n = len(W)
    if error_sd <= 0:
        raise ValueError(f"error_sd must be > 0, got {error_sd}.")

    if bandwidth is None:
        bandwidth = 0.9 * np.std(W) * n ** (-1.0 / 5.0)

    x_min, x_max = W.min() - 3 * np.std(W), W.max() + 3 * np.std(W)
    x_grid = np.linspace(x_min, x_max, grid_points)

    t_max = 1.0 / bandwidth
    m = 1024
    t = np.linspace(-t_max, t_max, m)
    dt = t[1] - t[0]

    phi_w = np.mean(np.exp(1j * np.outer(t, W)), axis=1)

    if error_type == "laplace":
        phi_u = 1.0 / (1.0 + (error_sd * t) ** 2)
    else:
        phi_u = np.exp(-0.5 * (error_sd * t) ** 2)

    phi_u = np.where(np.abs(phi_u) < 1e-10, 1e-10, phi_u)

    sinc_kernel = np.sinc(t * bandwidth / np.pi)

    integrand_base = phi_w / phi_u * sinc_kernel

    density = np.empty(grid_points)
    for j, x in enumerate(x_grid):
        val = np.real(np.sum(integrand_base * np.exp(-1j * t * x)) * dt)
        density[j] = val / (2 * np.pi)

    density = np.maximum(density, 0.0)
    area = np.trapezoid(density, x_grid)
    if area > 0:
        density /= area

    return {
        "x_grid": x_grid,
        "density": density,
        "bandwidth": bandwidth,
        "n": n,
        "method": "FourierDeconvolution",
    }


dcnvl_fn = dcnvl


def cheatsheet() -> str:
    return "dcnvl(W) -> Deconvolution density estimation (Fan 1991; Horowitz 2009)."
