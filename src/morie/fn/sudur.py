"""Duration of substance use analysis."""

import numpy as np

from ._containers import DescriptiveResult


def substance_duration(
    durations: list | np.ndarray,
) -> DescriptiveResult:
    """Analyse duration of substance use in years.

    Parameters
    ----------
    durations : array-like
        Duration of use in years for each individual.

    Returns
    -------
    DescriptiveResult
    """
    d = np.asarray(durations, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        raise ValueError("No valid durations provided")

    return DescriptiveResult(
        name="substance_duration",
        value=float(np.mean(d)),
        extra={
            "median": float(np.median(d)),
            "std": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
            "min": float(np.min(d)),
            "max": float(np.max(d)),
            "n": len(d),
            "pct_chronic_5yr": float(np.mean(d >= 5) * 100),
            "pct_chronic_10yr": float(np.mean(d >= 10) * 100),
        },
    )


sudur = substance_duration


def cheatsheet() -> str:
    return "substance_duration({}) -> Duration of substance use analysis."
