"""Signal stationarity test."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "When I let go of what I am, I become what I might be. -- Lao Tzu"


def stationarity_test(x, n_segments=5, **kwargs) -> DescriptiveResult:
    """Test signal stationarity via segment mean/variance comparison.

    Splits the signal into segments and checks whether mean and
    variance remain approximately constant across segments.

    Parameters
    ----------
    x : array-like
        Input signal.
    n_segments : int
        Number of segments.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if len(x) < n_segments * 2:
        raise ValueError("Signal too short for requested segments.")
    seg_len = len(x) // n_segments
    means = [float(np.mean(x[i * seg_len : (i + 1) * seg_len])) for i in range(n_segments)]
    variances = [float(np.var(x[i * seg_len : (i + 1) * seg_len])) for i in range(n_segments)]
    mean_spread = float(np.std(means))
    var_spread = float(np.std(variances))
    overall_std = float(np.std(x))
    stationary = bool(mean_spread < 0.1 * overall_std) if overall_std > 0 else True
    return DescriptiveResult(
        name="stationarity_test",
        value=mean_spread,
        extra={
            "segment_means": means,
            "segment_variances": variances,
            "mean_spread": mean_spread,
            "var_spread": var_spread,
            "stationary": stationary,
            "n_segments": n_segments,
        },
    )


stner = stationarity_test


def cheatsheet() -> str:
    return "stationarity_test({}) -> Signal stationarity test."
