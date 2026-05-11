# morie.fn — function file (hadesllm/morie)
"""Per-road-segment crash rate."""

from __future__ import annotations

import pandas as pd

from morie.fn._containers import DescriptiveResult


def mto_road_segment(
    df: pd.DataFrame,
    *,
    segment_col: str = "segment_id",
    crash_col: str = "n_crashes",
    traffic_col: str = "aadt",
) -> DescriptiveResult:
    """Compute crash rate per road segment normalised by AADT.

    Parameters
    ----------
    df : DataFrame
    segment_col, crash_col, traffic_col : str

    Returns
    -------
    DescriptiveResult
    """
    for c in [segment_col, crash_col, traffic_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found")
    df2 = df.copy()
    df2["_rate"] = df2[crash_col] / df2[traffic_col].replace(0, float("nan")) * 1e6
    grouped = df2.groupby(segment_col)["_rate"].mean()
    return DescriptiveResult(
        name="road_segment_rate",
        value=float(grouped.mean()),
        extra={"segment_rates": grouped.to_dict(), "n_segments": len(grouped)},
    )


mtord = mto_road_segment


def cheatsheet() -> str:
    return "mto_road_segment({}) -> Per-road-segment crash rate."
