# morie.fn -- function file (hadesllm/morie)
"""Mean change-point detection via CUSUM. 'In time, the suffering of your people will persuade you.' -- Count Dooku"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mean_changepoint(y: np.ndarray, method: str = "cusum") -> DescriptiveResult:
    r"""
    Detect a single mean shift in a time series using CUSUM.

    Computes the cumulative sum of deviations from the grand mean
    and locates the point of maximum absolute departure:

    .. math::

        S_k = \\sum_{i=1}^{k} (y_i - \\bar{y}), \\quad
        \\hat\\tau = \\arg\\max_k |S_k|

    :param y: 1-D time series.
    :type y: numpy.ndarray
    :param method: Currently only ``"cusum"``. Default ``"cusum"``.
    :type method: str
    :return: DescriptiveResult with change point and CUSUM statistic.
    :rtype: DescriptiveResult

    References
    ----------
    Page E.S. (1954). Continuous inspection schemes. *Biometrika*,
    41(1/2), 100-115.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if n < 4:
        raise ValueError(f"Need >= 4 observations, got {n}.")
    grand_mean = np.mean(y)
    cusum = np.cumsum(y - grand_mean)
    tau = int(np.argmax(np.abs(cusum)))
    max_stat = float(np.abs(cusum[tau]))
    mean_before = float(np.mean(y[: tau + 1]))
    mean_after = float(np.mean(y[tau + 1 :])) if tau + 1 < n else mean_before
    return DescriptiveResult(
        name="mean_changepoint",
        value=tau,
        extra={
            "changepoint": tau,
            "cusum_statistic": max_stat,
            "mean_before": mean_before,
            "mean_after": mean_after,
            "method": method,
        },
    )


cpmnr = mean_changepoint


def cheatsheet() -> str:
    return "mean_changepoint({}) -> Mean change-point detection via CUSUM. 'In time, the sufferi"
