# morie.fn -- function file (hadesllm/morie)
"""Kernel-smoothed sign test."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["ksgnr"]


def ksgnr(data: np.ndarray, cdf=None, *, mu0: float = 0.0, bw: float | None = None) -> dict:
    r"""
    Kernel-smoothed sign test for the median.

    Tests :math:`H_0: \text{median} = \mu_0` using a smoothed version
    of the sign statistic:

    .. math::

        S_h = \frac{1}{n} \sum_{i=1}^{n}
        \Phi\!\left(\frac{X_i - \mu_0}{h}\right) - \frac{1}{2}

    where :math:`\Phi` is the standard normal CDF and the smoothing
    makes the statistic asymptotically normal under the null.

    Parameters
    ----------
    data : np.ndarray
        1-d observations.
    mu0 : float
        Hypothesized median. Default 0.
    bw : float or None
        Smoothing bandwidth. If None, uses n^{-1/3} scaling.

    Returns
    -------
    dict
        ``statistic``, ``p_value``, ``mu0``, ``bw``.

    References
    ----------
    Hutton, J. L. & Nelson, P. I. (1986). Interrelationships between a
        kernel-type sign test and other nonparametric tests. *Biometrika*,
        73(3), 723-726.
    """
    from scipy.stats import norm

    data = np.asarray(data, dtype=float).ravel()
    n = data.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 observations.")

    if bw is None:
        bw = max(np.std(data, ddof=1) * n ** (-1.0 / 3), 1e-10)
    if bw <= 0:
        raise ValueError(f"bw must be positive, got {bw}.")

    smoothed = norm.cdf((data - mu0) / bw)
    s_bar = np.mean(smoothed) - 0.5

    f0 = 1.0 / (bw * np.sqrt(2 * np.pi))
    var_s = 0.25 / n
    z = s_bar / np.sqrt(var_s)

    p_value = float(2 * norm.sf(np.abs(z)))

    return RichResult(payload={"statistic": float(z), "p_value": p_value, "mu0": mu0, "bw": bw})


def cheatsheet() -> str:
    return "ksgnr({data}) -> Kernel-smoothed sign test for the median."
