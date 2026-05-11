# morie.fn — function file (hadesllm/morie)
"""Occupational exposure assessment (TWA vs OEL)."""

import numpy as np

from ._containers import DescriptiveResult


def occupational_exposure(
    measurements: list | np.ndarray,
    oel: float,
) -> DescriptiveResult:
    """Assess occupational exposure relative to occupational exposure limit.

    Parameters
    ----------
    measurements : array-like
        Exposure measurements.
    oel : float
        Occupational exposure limit.

    Returns
    -------
    DescriptiveResult
    """
    m = np.asarray(measurements, dtype=float)
    if len(m) == 0:
        raise ValueError("No measurements provided")
    if oel <= 0:
        raise ValueError("OEL must be positive")

    twa = float(np.mean(m))
    pct_oel = twa / oel * 100
    n_exceed = int(np.sum(m > oel))

    if pct_oel < 10:
        category = "negligible"
    elif pct_oel < 50:
        category = "low"
    elif pct_oel < 100:
        category = "moderate"
    else:
        category = "overexposed"

    return DescriptiveResult(
        name="occupational_exposure",
        value=twa,
        extra={
            "oel": float(oel),
            "pct_oel": float(pct_oel),
            "category": category,
            "n_exceed": n_exceed,
            "max": float(np.max(m)),
            "n": len(m),
        },
    )


occex = occupational_exposure


def cheatsheet() -> str:
    return "occupational_exposure({}) -> Occupational exposure assessment (TWA vs OEL)."
