# morie.fn -- function file (hadesllm/morie)
"""Edgeworth expansion for kernel density statistics."""

from __future__ import annotations

import numpy as np

__all__ = ["hedgw"]


def hedgw(
    data: np.ndarray,
    x0: float,
    *,
    bw: float | None = None,
) -> dict:
    r"""
    Edgeworth expansion for the kernel density estimator at a point.

    Provides a higher-order normal approximation to the distribution
    of :math:`\hat{f}_h(x_0)`:

    .. math::

        P\!\left(\frac{\hat{f}_h(x_0) - E[\hat{f}_h(x_0)]}
        {\sqrt{\mathrm{Var}[\hat{f}_h(x_0)]}} \le z\right)
        \approx \Phi(z) + \frac{\hat\kappa_3}{6\sqrt{n}}\,(z^2 - 1)\phi(z)

    where :math:`\hat\kappa_3` is the estimated skewness of the kernel
    contributions.

    Parameters
    ----------
    data : np.ndarray
        1-d observations.
    x0 : float
        Point at which the KDE is evaluated.
    bw : float or None
        Bandwidth. Silverman's rule if None.

    Returns
    -------
    dict
        ``f_hat``, ``bias_term``, ``variance``, ``skewness_correction``,
        ``ci_lower``, ``ci_upper`` (95% Edgeworth-corrected CI), ``bw``.

    References
    ----------
    Hall, P. (1992). *The Bootstrap and Edgeworth Expansion*. Springer.
        Chapter 3.
    """

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 observations.")

    sigma = np.std(data, ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    s = min(sigma, iqr / 1.349) if iqr > 0 else sigma
    if bw is None:
        bw = max(0.9 * s * n ** (-0.2), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    u = (x0 - data) / bw
    k_vals = np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    f_hat = float(np.mean(k_vals) / bw)

    rk = 1.0 / (2.0 * np.sqrt(np.pi))
    variance = rk * f_hat / (n * bw) if f_hat > 0 else 1e-20

    k2_vals = np.exp(-0.5 * u ** 2 / 2) / np.sqrt(4 * np.pi)
    f2_hat = float(np.mean(k2_vals) / (bw * np.sqrt(2)))

    mu3_k = np.mean(k_vals ** 3) / bw ** 3
    mu2_k = np.mean(k_vals ** 2) / bw ** 2
    if mu2_k > 0:
        kappa3 = mu3_k / mu2_k ** 1.5
    else:
        kappa3 = 0.0

    bias_term = 0.5 * bw ** 2 * f2_hat

    se = np.sqrt(variance)
    z_alpha = 1.96
    correction = kappa3 / (6.0 * np.sqrt(n))

    ci_lower = f_hat - z_alpha * se - correction * se
    ci_upper = f_hat + z_alpha * se + correction * se

    return {
        "f_hat": f_hat,
        "bias_term": float(bias_term),
        "variance": float(variance),
        "skewness_correction": float(kappa3),
        "ci_lower": float(max(ci_lower, 0.0)),
        "ci_upper": float(ci_upper),
        "bw": bw,
    }


def cheatsheet() -> str:
    return "hedgw({data}, x0) -> Edgeworth expansion for KDE at a point."
