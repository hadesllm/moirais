# morie.fn — function file (hadesllm/morie)
"""Prohorov metric between empirical distributions."""

from __future__ import annotations

import numpy as np


def prohorov_metric(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    r"""
    Prohorov (Prokhorov) metric between two empirical distributions.

    The Prohorov metric metrizes weak convergence of probability measures.
    For measures :math:`P` and :math:`Q` on :math:`(\mathbb{R}, \mathcal{B})`:

    .. math::

        \pi(P, Q) = \inf\!\left\{\varepsilon > 0 :
        P(A) \le Q(A^\varepsilon) + \varepsilon
        \;\text{for all Borel } A\right\}

    where :math:`A^\varepsilon = \{x : d(x, A) < \varepsilon\}`.

    For empirical measures the Prohorov metric satisfies:

    .. math::

        D_n^2 \le \pi(P_n, Q_m) \le D_n

    where :math:`D_n` is the Kolmogorov-Smirnov statistic. This function
    computes exact Prohorov distance via binary search on epsilon.

    :param x: 1-D array, sample from first distribution.
    :param y: 1-D array, sample from second distribution.
    :return: dict with ``prohorov`` (metric value), ``ks_statistic``,
        ``ks_lower`` (D^2), ``n``, ``m``.
    :raises ValueError: If either array is empty.

    References
    ----------
    Prohorov, Yu.V. (1956). Convergence of random processes and limit
        theorems in probability theory. *Theory Probab. Appl.*, 1(2), 157--214.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Ch. 7. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("Both x and y must be non-empty.")

    n, m = x.size, y.size
    combined = np.sort(np.concatenate([x, y]))

    ecdf_x = np.searchsorted(np.sort(x), combined, side="right") / n
    ecdf_y = np.searchsorted(np.sort(y), combined, side="right") / m

    ks_stat = float(np.max(np.abs(ecdf_x - ecdf_y)))

    lo, hi = ks_stat**2, ks_stat
    for _ in range(100):
        eps = (lo + hi) / 2.0
        xs = np.sort(x)
        ys = np.sort(y)

        ok = True
        for t in combined:
            fx = np.searchsorted(xs, t, side="right") / n
            fy_eps = np.searchsorted(ys, t + eps, side="right") / m
            if fx > fy_eps + eps + 1e-12:
                ok = False
                break
            fy = np.searchsorted(ys, t, side="right") / m
            fx_eps = np.searchsorted(xs, t + eps, side="right") / n
            if fy > fx_eps + eps + 1e-12:
                ok = False
                break

        if ok:
            hi = eps
        else:
            lo = eps

        if hi - lo < 1e-10:
            break

    return {
        "prohorov": float(hi),
        "ks_statistic": ks_stat,
        "ks_lower": ks_stat**2,
        "n": n,
        "m": m,
        "method": "Prohorov metric",
    }


prohv = prohorov_metric


def cheatsheet() -> str:
    return "prohorov_metric({x}, {y}) -> Prohorov metric between distributions."
