# moirais.fn — function file (hadesllm/moirais)
"""Epidemic curve (epi curve) construction."""

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult


def epidemic_curve(
    dates: np.ndarray,
    bin_width: str = "week",
) -> DescriptiveResult:
    """Construct an epidemic curve by binning event dates.

    Parameters
    ----------
    dates : array-like
        Array of event dates (datetime64, str, or pd.Timestamp).
    bin_width : str, default "week"
        Bin width: "day", "week", or "month".

    Returns
    -------
    DescriptiveResult
        value = DataFrame with columns ['bin_start', 'count'].

    References
    ----------
    CDC (2012). Principles of Epidemiology in Public Health Practice.
    3rd ed. Lesson 6: Investigating an Outbreak.
    """
    dates_s = pd.to_datetime(np.asarray(dates))
    n_total = len(dates_s)

    freq_map = {"day": "D", "week": "W-MON", "month": "MS"}
    freq = freq_map.get(bin_width)
    if freq is None:
        raise ValueError(f"bin_width must be one of {list(freq_map.keys())}")

    s = pd.Series(1, index=dates_s)
    grouped = s.resample(freq).sum().fillna(0).astype(int)
    df = pd.DataFrame({"bin_start": grouped.index, "count": grouped.values})

    return DescriptiveResult(
        name="Epidemic curve",
        value=df,
        extra={"bin_width": bin_width, "n_total": n_total},
    )


epi_c = epidemic_curve


def cheatsheet() -> str:
    return "epidemic_curve({}) -> Epidemic curve (epi curve) construction."
