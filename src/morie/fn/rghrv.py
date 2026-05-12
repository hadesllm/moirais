# morie.fn — function file (hadesllm/morie)
"""Heart rate variability (time-domain) — Rangayyan Ch 6."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_hrv"]


def rangayyan_hrv(rr_ms):
    """Time-domain HRV indices from RR-interval series.

    Parameters
    ----------
    rr_ms : array-like
        Sequence of consecutive RR (NN) intervals in milliseconds.

    Returns
    -------
    RichResult with keys

      - ``meanNN``    mean of NN intervals (ms)
      - ``SDNN``      standard deviation of NN intervals (ms)
      - ``RMSSD``     root-mean-square of successive differences (ms)
      - ``pNN50``     percentage of pairs of adjacent NN differing > 50 ms
      - ``heart_rate_bpm``  60000 / meanNN

    Standard adult resting HRV norms (Task Force 1996):
    SDNN ≥ 50 ms, RMSSD 20–50 ms.

    References
    ----------
    Rangayyan Ch 6.  Task Force of the ESC/NASPE (1996), Circulation 93:1043.
    """
    rr = np.asarray(rr_ms, dtype=float).ravel()
    n = rr.size
    if n < 2:
        raise ValueError("Need at least 2 RR intervals.")
    mean_nn = float(rr.mean())
    sdnn = float(rr.std(ddof=1))
    d = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(d ** 2)))
    pnn50 = float(100.0 * np.mean(np.abs(d) > 50.0))
    hr = 60000.0 / mean_nn if mean_nn > 0 else float("nan")
    warnings = []
    if sdnn < 50:
        warnings.append(
            "SDNN below 50 ms — reduced overall variability (Task Force 1996 reference)."
        )
    res = RichResult(
        title="Heart Rate Variability (time domain)",
        summary_lines=[
            ("N intervals", n),
            ("meanNN (ms)", mean_nn),
            ("SDNN (ms)", sdnn),
            ("RMSSD (ms)", rmssd),
            ("pNN50 (%)", pnn50),
            ("Heart rate (bpm)", hr),
        ],
        warnings=warnings,
        interpretation=(
            f"Mean HR {hr:.1f} bpm; SDNN {sdnn:.2f} ms; RMSSD {rmssd:.2f} ms."
        ),
        payload={
            "meanNN": mean_nn,
            "SDNN": sdnn,
            "RMSSD": rmssd,
            "pNN50": pnn50,
            "heart_rate_bpm": hr,
            "n": n,
        },
    )
    return with_describe_pointer(res, "rghrv")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> rr = 800 + rng.normal(0, 40, 200)
# >>> r = rangayyan_hrv(rr)
# >>> 700 < r["meanNN"] < 900
# True


def cheatsheet():
    return "rghrv: time-domain HRV (SDNN, RMSSD, pNN50) — Rangayyan Ch 6"
