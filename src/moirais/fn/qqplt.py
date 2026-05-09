# moirais.fn — function file (hadesllm/moirais)
"""Quantile-quantile plot data for distribution comparison."""

import numpy as np

from ._containers import DescriptiveResult


def qq_data(x, distribution="norm"):
    """
    Compute Q-Q plot data comparing sample quantiles to theoretical.

    Returns paired theoretical and sample quantiles plus the
    reference line parameters for assessing distributional fit.

    :param x: (n,) numeric data.
    :param distribution: Theoretical distribution ('norm', 't', 'uniform', 'exp').
    :return: DescriptiveResult with theoretical and sample quantiles.

    References
    ----------
    Wilk MB & Gnanadesikan R (1968). Probability Plotting Methods
    for the Analysis of Data. Biometrika 55(1):1-17.
    """
    from scipy import stats as sp_stats

    arr = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = len(arr)

    probs = (np.arange(1, n + 1) - 0.5) / n

    dist_map = {
        "norm": sp_stats.norm,
        "t": sp_stats.t(df=n - 1),
        "uniform": sp_stats.uniform,
        "exp": sp_stats.expon,
    }
    dist = dist_map.get(distribution, sp_stats.norm)
    theoretical = dist.ppf(probs)

    q25_idx = max(0, int(0.25 * n) - 1)
    q75_idx = min(n - 1, int(0.75 * n))
    slope = (
        (arr[q75_idx] - arr[q25_idx]) / (theoretical[q75_idx] - theoretical[q25_idx])
        if theoretical[q75_idx] != theoretical[q25_idx]
        else 1.0
    )
    intercept = arr[q25_idx] - slope * theoretical[q25_idx]

    correlation = float(np.corrcoef(theoretical, arr)[0, 1])

    return DescriptiveResult(
        name="qq_data",
        value=float(correlation),
        extra={
            "theoretical_quantiles": theoretical.tolist(),
            "sample_quantiles": arr.tolist(),
            "distribution": distribution,
            "n": n,
            "qq_correlation": float(correlation),
            "reference_slope": float(slope),
            "reference_intercept": float(intercept),
        },
    )


def cheatsheet() -> str:
    return "qq_data({}) -> Quantile-quantile plot data for distribution comparison."
