# morie.fn — function file (hadesllm/morie)
"""Bandwidth selection via Sheather-Jones plug-in."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kbwpi"]


def kbwpi(data: np.ndarray) -> dict:
    r"""
    Sheather-Jones plug-in bandwidth selector.

    Estimates the AMISE-optimal bandwidth by iteratively solving:

    .. math::

        h = \left[\frac{R(K)}{n\,\mu_2(K)^2\,\hat{\sigma}_K^{(4)}(g)}\right]^{1/5}

    where :math:`\hat{\sigma}_K^{(4)}` is a kernel estimate of the
    integrated squared second derivative of *f*, evaluated at a pilot
    bandwidth *g*.

    Parameters
    ----------
    data : np.ndarray
        1-d array of observations.

    Returns
    -------
    dict
        ``bw_opt``, ``n``.

    References
    ----------
    Sheather, S. J. & Jones, M. C. (1991). A reliable data-based bandwidth
        selection method for kernel density estimation. *JRSS-B*,
        53(3), 683-690.
    """
    from scipy.stats import norm

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    s = max(s, 1e-10)

    diffs = data[:, None] - data[None, :]

    def _phi_r(r, g):
        """Kernel estimate of int f^(r)(x)^2 dx using Gaussian kernel."""
        coeff = (-1) ** (r // 2)
        vals = norm.pdf(diffs / g, 0, 1)
        from scipy.special import hermite
        H = hermite(r)
        vals = vals * H(diffs / g) / (g ** (r + 1))
        return coeff * vals.sum() / (n * (n - 1))

    lam = np.subtract(*np.percentile(data, [75, 25]))
    lam = max(lam, sigma)

    a = (8 * np.sqrt(np.pi) * 15 / 3.0) ** 0.2
    g6 = a * s * n ** (-1.0 / 7)

    phi6 = _phi_r(6, g6)
    if abs(phi6) < 1e-30:
        phi6 = -15.0 / (16 * np.sqrt(np.pi) * s ** 7)

    g4 = (-6.0 / (np.sqrt(2 * np.pi) * phi6 * n)) ** (1.0 / 7)
    g4 = max(g4, s * 0.01)

    phi4 = _phi_r(4, g4)
    if abs(phi4) < 1e-30:
        phi4 = 3.0 / (8 * np.sqrt(np.pi) * s ** 5)

    rk = 1.0 / (2 * np.sqrt(np.pi))
    ratio = rk / (n * phi4)
    if ratio <= 0:
        bw_opt = 0.9 * s * n ** (-0.2)
    else:
        bw_opt = ratio ** 0.2
    bw_opt = max(bw_opt, 1e-10)

    return RichResult(payload={"bw_opt": float(bw_opt), "n": n})


def cheatsheet() -> str:
    return "kbwpi({data}) -> Sheather-Jones plug-in bandwidth."
