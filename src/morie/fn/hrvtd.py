# morie.fn — function file (hadesllm/morie)
"""HRV time-domain metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hrv_time_domain(rr: np.ndarray) -> DescriptiveResult:
    """HRV time-domain analysis (SDNN, RMSSD, pNN50).

    :param rr: 1-D array of RR intervals in milliseconds.
    :return: DescriptiveResult with metrics in ``extra``.
    """
    rr = np.asarray(rr, dtype=float).ravel()
    if len(rr) < 2:
        return DescriptiveResult(name="hrv_time_domain", value=float("nan"))

    sdnn = float(np.std(rr, ddof=1))
    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr**2)))
    pnn50 = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100)
    mean_rr = float(np.mean(rr))
    mean_hr = 60000.0 / mean_rr if mean_rr > 0 else float("nan")
    tri_idx = len(rr) / (np.max(np.histogram(rr, bins=128)[0]) + 1e-12)

    return DescriptiveResult(
        name="hrv_time_domain",
        value=sdnn,
        extra={
            "sdnn": sdnn,
            "rmssd": rmssd,
            "pnn50": pnn50,
            "mean_rr": mean_rr,
            "mean_hr": float(mean_hr),
            "hrv_triangular_index": float(tri_idx),
            "n_intervals": len(rr),
        },
    )


hrvtd = hrv_time_domain


def cheatsheet() -> str:
    return "hrv_time_domain({}) -> HRV time-domain metrics."
