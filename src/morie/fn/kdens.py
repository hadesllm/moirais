# morie.fn -- function file (hadesllm/morie)
"""Kernel density estimation."""

from __future__ import annotations

import numpy as np


def kdens(
    x: np.ndarray,
    x_eval: np.ndarray | None = None,
    *,
    bandwidth: float | None = None,
    kernel: str = "gaussian",
    n_grid: int = 256,
) -> dict:
    r"""
    Kernel density estimation.

    Estimates the probability density function :math:`f(x)` via:

    .. math::

        \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\!\left(\frac{x - X_i}{h}\right)

    where :math:`K` is a kernel function and :math:`h` is the bandwidth.

    Parameters
    ----------
    x : np.ndarray
        1-d array of observed values (n,).
    x_eval : np.ndarray or None
        Grid points at which to evaluate the density. If None, an
        equally-spaced grid of ``n_grid`` points spanning the data range
        (extended by 3 bandwidths) is used.
    bandwidth : float or None
        Kernel bandwidth. If None, uses Silverman's rule of thumb.
    kernel : str
        Kernel function: ``'gaussian'``, ``'epanechnikov'``, or ``'uniform'``.
    n_grid : int
        Number of grid points when ``x_eval`` is None. Default 256.

    Returns
    -------
    dict
        Keys: ``x_eval`` (grid), ``density`` (estimated density values),
        ``bandwidth``, ``kernel``, ``n_obs``.

    Raises
    ------
    ValueError
        If unknown kernel or insufficient data.

    References
    ----------
    Silverman, B. W. (1986). Density Estimation for Statistics and Data
        Analysis. Chapman & Hall.
    Horowitz, J. L. (2009). Semiparametric and Nonparametric Methods in
        Econometrics. Springer. Chapter 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    valid_kernels = {"gaussian", "epanechnikov", "uniform"}
    if kernel not in valid_kernels:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose from {valid_kernels}.")

    if bandwidth is None:
        from morie.fn.nwker import _silverman_bw
        bandwidth = _silverman_bw(x)

    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}.")

    if x_eval is None:
        lo = x.min() - 3 * bandwidth
        hi = x.max() + 3 * bandwidth
        x_eval = np.linspace(lo, hi, n_grid)
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()

    from morie.fn.nwker import _get_kernel
    k_fn = _get_kernel(kernel)

    u = (x_eval[:, None] - x[None, :]) / bandwidth
    density = k_fn(u).mean(axis=1) / bandwidth

    return {
        "x_eval": x_eval,
        "density": density,
        "bandwidth": bandwidth,
        "kernel": kernel,
        "n_obs": n,
    }


kdens_fn = kdens


def cheatsheet() -> str:
    return "kdens({x}) -> Kernel density estimation (Gaussian/Epanechnikov/uniform)."
