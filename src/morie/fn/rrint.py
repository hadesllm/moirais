# morie.fn -- function file (hadesllm/morie)
"""RR interval series from R-peak indices."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def rr_intervals(
    r_peaks: np.ndarray,
    fs: float,
) -> DescriptiveResult:
    """Compute RR interval series from R-peak sample indices.

    :param r_peaks: 1-D array of R-peak sample indices.
    :param fs: Sampling frequency in Hz.
    :return: DescriptiveResult with RR intervals (ms) in ``extra["rr_ms"]``.
    """
    r_peaks = np.asarray(r_peaks, dtype=int).ravel()
    if len(r_peaks) < 2:
        return DescriptiveResult(
            name="rr_intervals",
            value=float("nan"),
            extra={"rr_ms": np.array([])},
        )

    rr_samples = np.diff(r_peaks)
    rr_ms = rr_samples / fs * 1000.0

    return DescriptiveResult(
        name="rr_intervals",
        value=float(np.mean(rr_ms)),
        extra={
            "rr_ms": rr_ms,
            "mean_rr": float(np.mean(rr_ms)),
            "std_rr": float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0,
            "n_intervals": len(rr_ms),
        },
    )


rrint = rr_intervals


def cheatsheet() -> str:
    return "rr_intervals({}) -> RR interval series from R-peak indices."
