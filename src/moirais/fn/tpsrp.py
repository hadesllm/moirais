"""Police report summary statistics."""

from __future__ import annotations

import pandas as pd

from moirais.fn._containers import DescriptiveResult


def tps_report_analysis(
    df: pd.DataFrame,
    *,
    offense_col: str = "offense_type",
    date_col: str = "report_date",
) -> DescriptiveResult:
    """Summarise police reports by offense type.

    Computes frequency counts, proportions, and temporal range
    for each offense category in a police report dataset.

    Parameters
    ----------
    df : DataFrame
        Police report data.
    offense_col : str
        Column containing offense type labels.
    date_col : str
        Column containing report dates.

    Returns
    -------
    DescriptiveResult
        Offense frequency summary with temporal range.
    """
    if offense_col not in df.columns:
        raise ValueError(f"Column '{offense_col}' not found in DataFrame")
    counts = df[offense_col].value_counts()
    props = counts / counts.sum()
    extra = {
        "counts": counts.to_dict(),
        "proportions": props.to_dict(),
        "n_offenses": int(counts.sum()),
        "n_types": len(counts),
    }
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            extra["date_min"] = str(dates.min())
            extra["date_max"] = str(dates.max())
    return DescriptiveResult(name="tps_report_analysis", value=len(counts), extra=extra)


tpsrp = tps_report_analysis


def cheatsheet() -> str:
    return "tps_report_analysis({}) -> Police report summary statistics."
