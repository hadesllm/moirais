# morie.fn -- function file (hadesllm/morie)
"""Heart rate from RR intervals."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def heart_rate_from_rr(rr_intervals) -> DescriptiveResult:
    """Convert RR intervals (seconds) to heart rate (BPM).

    Parameters
    ----------
    rr_intervals : array-like
        RR intervals in seconds.

    Returns
    -------
    DescriptiveResult
    """
    from morie._detection import heart_rate_from_rr as _hr

    rr = np.asarray(rr_intervals, dtype=float)
    hr = _hr(rr)
    return DescriptiveResult(
        name="heart_rate_from_rr",
        value=float(np.nanmean(hr)),
        extra={"heart_rates": hr, "mean_hr": float(np.nanmean(hr))},
    )


hrrr = heart_rate_from_rr


def cheatsheet() -> str:
    return "heart_rate_from_rr({}) -> Heart rate from RR intervals."
