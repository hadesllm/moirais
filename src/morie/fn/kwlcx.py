# morie.fn -- function file (hadesllm/morie)
"""Kernel-smoothed Wilcoxon signed-rank test."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["kwlcx"]


def kwlcx(data: np.ndarray, cdf=None, *, mu0: float = 0.0, bw: float | None = None) -> dict:
    r"""
    Kernel-smoothed Wilcoxon signed-rank test.

    Tests :math:`H_0: \text{median} = \mu_0` using a smoothed rank
    statistic:

    .. math::

        W_h = \frac{1}{\binom{n}{2}} \sum_{i<j}
        \Phi\!\left(\frac{(X_i - \mu_0) + (X_j - \mu_0)}{2h}\right)
        - \frac{1}{2}

    Parameters
    ----------
    data : np.ndarray
        1-d observations.
    mu0 : float
        Hypothesized location. Default 0.
    bw : float or None
        Smoothing bandwidth. If None, n^{-1/3} scaling.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``mu0``, ``bw``.

    References
    ----------
    Hettmansperger, T. P. (1984). *Statistical Inference Based on Ranks*.
        Wiley. Chapter 3.
    """
    from scipy.stats import norm

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    if bw is None:
        bw = max(np.std(data, ddof=1) * n ** (-0.2), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    d = data - mu0
    walsh = (d[:, None] + d[None, :]) / 2.0

    psi = norm.cdf(walsh / bw)
    theta = np.mean(psi)

    g1 = np.mean(psi, axis=1)
    zeta1 = np.var(g1, ddof=1)
    var_theta = 4.0 * zeta1 / n
    var_theta = max(var_theta, 1e-20)

    w_bar = theta - 0.5
    z = w_bar / np.sqrt(var_theta)

    p_value = float(2 * norm.sf(np.abs(z)))

    return RichResult(payload={"statistic": float(z), "p_value": p_value, "mu0": mu0, "bw": bw})


def cheatsheet() -> str:
    return "kwlcx({data}) -> Kernel-smoothed Wilcoxon signed-rank test."
