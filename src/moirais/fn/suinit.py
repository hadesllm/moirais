"""Age of substance use initiation analysis."""

import numpy as np

from ._containers import DescriptiveResult


def initiation_age(
    ages_of_first_use: list | np.ndarray,
) -> DescriptiveResult:
    """Analyse distribution of age at first substance use.

    Parameters
    ----------
    ages_of_first_use : array-like
        Ages at which individuals first used the substance.

    Returns
    -------
    DescriptiveResult
    """
    a = np.asarray(ages_of_first_use, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        raise ValueError("No valid ages provided")

    return DescriptiveResult(
        name="initiation_age",
        value=float(np.median(a)),
        extra={
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "n": len(a),
            "pct_under_18": float(np.mean(a < 18) * 100),
        },
    )


suinit = initiation_age


def cheatsheet() -> str:
    return "initiation_age({}) -> Age of substance use initiation analysis."
