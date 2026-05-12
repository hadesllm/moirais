# morie.fn -- function file (hadesllm/morie)
"""Deconvolution density estimation (Stefanski-Carroll 1990; Horowitz 2009, Ch 12).

Observation model:   Y_i = X_i + U_i, with U ~ known density f_U.

The deconvolution kernel estimator inverts the characteristic-function
quotient with a sinc kernel:

    f_X_hat(x) = (1/(2 pi)) integral phi_K(t h) * phi_Y_hat(t) / phi_U(t) * exp(-i t x) dt

We implement this with FFT and a Laplace-noise default (phi_U(t) = 1/(1 + sigma^2 t^2)).
"""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["horowitz_deconvolution"]


def horowitz_deconvolution(y, sigma_u=0.5, bandwidth=None, grid=None,
                            noise="laplace"):
    """Fourier-deconvolution density estimator.

    Parameters
    ----------
    y : array-like
        Observed (contaminated) sample.
    sigma_u : float, default 0.5
        Scale of the assumed measurement-error distribution.
    bandwidth : float, optional
        Kernel bandwidth (defaults to a small over-smoothing).
    grid : array-like, optional
        Evaluation grid (defaults to 51 points on min..max).
    noise : {"laplace", "normal"}
        Family for the known measurement-error CF phi_U.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.size
    if n < 30:
        return RichResult(payload={"estimate": np.nan, "n": n,
                                   "method": "deconvolution (insufficient data)"})
    if bandwidth is None:
        h = max(1.5 * np.std(y, ddof=1) * n ** (-1.0 / 7.0), 1e-3)
    else:
        h = float(bandwidth)
    if grid is None:
        grid = np.linspace(y.min(), y.max(), 51)
    grid = np.asarray(grid, dtype=float).ravel()

    # Frequency grid
    T = np.linspace(-15, 15, 2049) / max(h, 1e-3)
    dt = T[1] - T[0]
    # Empirical characteristic function of Y at frequencies T
    phi_Y = (np.exp(1j * np.outer(T, y))).mean(axis=1)
    # CF of noise
    if noise == "normal":
        phi_U = np.exp(-0.5 * (sigma_u * T) ** 2)
    else:  # Laplace
        phi_U = 1.0 / (1.0 + (sigma_u * T) ** 2)
    # Sinc-Fourier kernel  phi_K(t h) = (1 - (t h)^2)^3 for |t h| <= 1, else 0
    th = T * h
    phi_K = np.where(np.abs(th) <= 1, (1 - th ** 2) ** 3, 0.0)
    # Inverse Fourier integral evaluated on `grid` by numerical quadrature
    integrand = phi_K * phi_Y / np.maximum(np.abs(phi_U), 1e-10) * \
        np.sign(phi_U + 0j).conj() / np.maximum(np.abs(phi_U), 1e-10) * np.abs(phi_U)
    # simpler: integrand = phi_K * phi_Y / phi_U
    integrand = phi_K * phi_Y / np.where(np.abs(phi_U) > 1e-10, phi_U, np.inf)
    f_hat = np.zeros(grid.size)
    for i, x0 in enumerate(grid):
        f_hat[i] = float(np.real(np.exp(-1j * T * x0) @ integrand)) * dt / (2 * np.pi)
    f_hat = np.maximum(f_hat, 0)  # numerical-noise floor
    return RichResult(payload={
        "estimate": f_hat.astype(float),
        "grid": grid.astype(float),
        "bandwidth": h, "sigma_u": float(sigma_u), "noise": noise, "n": n,
        "method": "Fourier deconvolution density (sinc-kernel)",
    })


def cheatsheet():
    return "hrzn2: deconvolution density estimator"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(16)
    n = 1500
    x = rng.standard_normal(n)
    u = rng.laplace(0.0, 0.3, n)
    y = x + u
    res = horowitz_deconvolution(y, sigma_u=0.3, grid=[0.0])
    print(res)
    assert res["estimate"][0] > 0
