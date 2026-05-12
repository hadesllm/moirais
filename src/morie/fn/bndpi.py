# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Plug-in bandwidth selection (Sheather-Jones)."""

from __future__ import annotations

import numpy as np


def bndpi(
    x: np.ndarray,
    *,
    kernel: str = "gaussian",
) -> dict:
    r"""
    Plug-in bandwidth selector (Sheather-Jones 1991).

    Estimates the AMISE-optimal bandwidth by plugging in a normal-reference
    estimate of the density's second derivative:

    .. math::

        h^* = \left(\frac{R(K)}{n \, \mu_2(K)^2 \, \sigma_K^2}\right)^{1/5}

    with iterative refinement for the roughness functional
    :math:`\int f''(x)^2 dx`.

    Parameters
    ----------
    x : np.ndarray
        Data vector (n,).
    kernel : str
        ``'gaussian'`` (only Gaussian supported for analytic SJ).

    Returns
    -------
    dict
        ``h_opt`` (optimal bandwidth), ``roughness`` (estimated
        :math:`\int f''^2`), ``n_obs``.

    References
    ----------
    Sheather, S. J. & Jones, M. C. (1991). A reliable data-based bandwidth
        selection method for kernel density estimation. JRSS-B, 53, 683-690.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")
    if kernel != "gaussian":
        raise ValueError("Sheather-Jones plug-in only supports Gaussian kernel.")

    sigma = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    s = min(sigma, iqr / 1.34) if iqr > 0 else sigma

    psi4_normal = 3.0 / (8.0 * np.sqrt(np.pi) * s**5)
    psi6_normal = -15.0 / (16.0 * np.sqrt(np.pi) * s**7)

    g1 = (-6.0 / (np.sqrt(2 * np.pi) * psi6_normal * n)) ** (1 / 7)
    g2 = (30.0 / (np.sqrt(2 * np.pi) * psi4_normal * n)) ** (1 / 9) if n > 1 else g1

    diff = x[:, None] - x[None, :]

    def gauss_deriv4(u):
        return (u**4 - 6 * u**2 + 3) * np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    def gauss_deriv6(u):
        return (u**6 - 15 * u**4 + 45 * u**2 - 15) * np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    u2 = diff / g2
    psi6_est = gauss_deriv6(u2).mean() / g2**7

    g1_refined = (-6.0 / (np.sqrt(2 * np.pi) * psi6_est * n)) ** (1 / 7) if abs(psi6_est) > 0 else g1

    u1 = diff / g1_refined
    psi4_est = gauss_deriv4(u1).mean() / g1_refined**5

    roughness = float(psi4_est) if psi4_est > 0 else float(psi4_normal)
    Rk = 1.0 / (2.0 * np.sqrt(np.pi))
    h_opt = (Rk / (n * roughness)) ** 0.2

    return {
        "h_opt": float(h_opt),
        "roughness": roughness,
        "n_obs": n,
    }


bndpi_fn = bndpi


def cheatsheet() -> str:
    return "bndpi({x}) -> Sheather-Jones plug-in bandwidth."
