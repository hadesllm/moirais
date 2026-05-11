# morie.fn — function file (hadesllm/morie)
"""Weather-related crash factor analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def mto_weather_factor(
    crashes: np.ndarray | list[int],
    weather_conditions: list[str],
) -> DescriptiveResult:
    """Analyse weather as a crash factor.

    Parameters
    ----------
    crashes : array-like
        Crash counts per condition entry.
    weather_conditions : list[str]
        Weather labels corresponding to each crash count.

    Returns
    -------
    DescriptiveResult
    """
    if len(crashes) != len(weather_conditions):
        raise ValueError("crashes and weather_conditions must be same length")
    s = pd.DataFrame({"w": weather_conditions, "c": np.asarray(crashes)})
    grouped = s.groupby("w")["c"].sum()
    total = grouped.sum()
    return DescriptiveResult(
        name="weather_crash_factor",
        value=float(total),
        extra={"by_weather": grouped.to_dict(), "proportions": (grouped / total).to_dict() if total > 0 else {}},
    )


mtowd = mto_weather_factor


def cheatsheet() -> str:
    return "mto_weather_factor({}) -> Weather-related crash factor analysis."
