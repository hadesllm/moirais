# morie.fn — function file (hadesllm/morie)
"""Heart rate variability -- time-domain metrics.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 11.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['hrvar']

_QUOTE = "Real knowledge is to know the extent of one's ignorance. — Confucius"


def hrvar(
    rr_intervals: np.ndarray,
    *,
    pnn_threshold: float = 0.05,
) -> DescriptiveResult:
    """Time-domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr_intervals : array-like
        RR intervals in seconds.
    pnn_threshold : float
        Threshold for pNNx (default 0.05 s = pNN50).

    Returns
    -------
    DescriptiveResult
        ``extra`` has SDNN, RMSSD, pNN50, mean_rr, mean_hr.
    """
    rr = np.asarray(rr_intervals, dtype=float).ravel()
    if len(rr) < 2:
        raise ValueError("Need at least 2 RR intervals.")

    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1))
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnnx = float(np.mean(np.abs(diffs) > pnn_threshold))
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else 0.0

    return DescriptiveResult(
        name="hrvar",
        value=sdnn,
        extra={
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnnx,
            "mean_rr": mean_rr,
            "mean_hr": mean_hr,
            "n_intervals": len(rr),
        },
    )


def cheatsheet() -> str:
    return "hrvar({}) -> HRV time-domain metrics."
