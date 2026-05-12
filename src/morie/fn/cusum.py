# morie.fn -- function file (hadesllm/morie)
"""CUSUM change-point detection for time series."""

import numpy as np


def cusum(
    series: np.ndarray,
    target_mean: float | None = None,
    threshold: float = 5.0,
    drift: float = 0.0,
) -> dict:
    r"""
    Cumulative Sum (CUSUM) change-point detection.

    Maintains upper and lower cumulative sum statistics:

    .. math::

        S^+_t = \\max(0,\\; S^+_{t-1} + (x_t - \\mu_0) - k)

        S^-_t = \\max(0,\\; S^-_{t-1} - (x_t - \\mu_0) - k)

    A change point is flagged when :math:`S^+_t` or :math:`S^-_t` exceeds
    the threshold *h*.

    :param series: 1-D array of observed values.
    :param target_mean: In-control mean :math:`\\mu_0`. If None, uses the
        overall sample mean. Default None.
    :param threshold: Decision interval *h*. Default 5.0.
    :param drift: Allowance parameter *k* (slack). Default 0.0.
    :return: dict with ``cusum_pos`` (array), ``cusum_neg`` (array),
        ``change_points`` (list of indices), ``target_mean``,
        ``threshold``.
    :raises ValueError: On empty series.

    References
    ----------
    Page, E. S. (1954). Continuous inspection schemes. *Biometrika*,
    41(1-2), 100-115.

    Montgomery, D. C. (2009). Introduction to Statistical Quality
    Control (6th ed.). Wiley.
    """
    y = np.asarray(series, dtype=float)
    if len(y) == 0:
        raise ValueError("Series must not be empty.")

    mu0 = target_mean if target_mean is not None else float(np.mean(y))
    n = len(y)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    change_points = []

    for t in range(n):
        diff = y[t] - mu0
        s_pos[t] = max(0.0, (s_pos[t - 1] if t > 0 else 0.0) + diff - drift)
        s_neg[t] = max(0.0, (s_neg[t - 1] if t > 0 else 0.0) - diff - drift)
        if s_pos[t] > threshold or s_neg[t] > threshold:
            change_points.append(t)
            # Reset after detection
            s_pos[t] = 0.0
            s_neg[t] = 0.0

    return {
        "cusum_pos": s_pos,
        "cusum_neg": s_neg,
        "change_points": change_points,
        "target_mean": mu0,
        "threshold": threshold,
    }


def cheatsheet() -> str:
    return "cusum({}) -> CUSUM change-point detection for time series."
