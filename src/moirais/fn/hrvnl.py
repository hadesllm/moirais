# moirais.fn — function file (hadesllm/moirais)
"""HRV nonlinear metrics."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hrv_nonlinear(rr: np.ndarray) -> DescriptiveResult:
    """HRV nonlinear analysis (Poincare SD1, SD2).

    :param rr: 1-D array of RR intervals in milliseconds.
    :return: DescriptiveResult with SD1, SD2 in ``extra``.
    """
    rr = np.asarray(rr, dtype=float).ravel()
    if len(rr) < 3:
        return DescriptiveResult(name="hrv_nonlinear", value=float("nan"))

    rr1 = rr[:-1]
    rr2 = rr[1:]
    diff = rr2 - rr1
    sd1 = float(np.std(diff, ddof=1) / np.sqrt(2))
    sd2 = float(np.sqrt(2 * np.std(rr, ddof=1) ** 2 - sd1**2))
    sd_ratio = sd1 / sd2 if sd2 > 0 else float("nan")

    return DescriptiveResult(
        name="hrv_nonlinear",
        value=sd1,
        extra={
            "sd1": sd1,
            "sd2": sd2,
            "sd1_sd2_ratio": sd_ratio,
            "n_intervals": len(rr),
        },
    )


hrvnl = hrv_nonlinear


def cheatsheet() -> str:
    return "hrv_nonlinear({}) -> HRV nonlinear metrics."
