# morie.fn — function file (hadesllm/morie)
"""R-R interval variability metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Who's the more foolish, the fool or the fool who follows him?"


def rr_variability(rr_intervals, **kwargs) -> DescriptiveResult:
    """Compute HRV time-domain metrics from R-R intervals.

    Parameters
    ----------
    rr_intervals : array-like
        R-R interval series in seconds.

    Returns
    -------
    DescriptiveResult
        Includes SDNN, RMSSD, pNN50, mean_rr.
    """
    rr = np.asarray(rr_intervals, dtype=float)
    if len(rr) < 2:
        return DescriptiveResult(
            name="rr_variability",
            value=0.0,
            extra={"sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0, "mean_rr": 0.0},
        )
    sdnn = float(np.std(rr, ddof=1))
    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr**2)))
    pnn50 = float(np.sum(np.abs(diff_rr) > 0.050) / len(diff_rr) * 100)
    mean_rr = float(np.mean(rr))
    return DescriptiveResult(
        name="rr_variability",
        value=sdnn,
        extra={
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "mean_rr": mean_rr,
            "n_intervals": len(rr),
        },
    )


rrvar = rr_variability


def cheatsheet() -> str:
    return "rr_variability({}) -> R-R interval variability metrics."
