# morie.fn -- function file (hadesllm/morie)
"""Empirical cumulative distribution function."""

import numpy as np

from ._containers import DescriptiveResult


def ecdf(x):
    """
    Compute the empirical cumulative distribution function.

    Returns sorted values and their cumulative probabilities.
    Useful for distribution visualization and two-sample comparisons.

    :param x: (n,) numeric data.
    :return: DescriptiveResult with sorted values and CDF values.

    References
    ----------
    van der Vaart AW (1998). Asymptotic Statistics. Cambridge.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = len(arr)
    sorted_x = np.sort(arr)
    cdf_values = np.arange(1, n + 1) / n

    q25 = float(np.percentile(arr, 25))
    q50 = float(np.percentile(arr, 50))
    q75 = float(np.percentile(arr, 75))

    dkw_band = np.sqrt(np.log(2 / 0.05) / (2 * n))

    return DescriptiveResult(
        name="ecdf",
        value=float(q50),
        extra={
            "sorted_values": sorted_x.tolist(),
            "cdf_values": cdf_values.tolist(),
            "n": n,
            "q25": q25,
            "median": q50,
            "q75": q75,
            "dkw_band_95": float(dkw_band),
        },
    )


def cheatsheet() -> str:
    return "ecdf({}) -> Empirical cumulative distribution function."
