# morie.fn — function file (hadesllm/morie)
"""Zero-crossing rate — Rangayyan Ch 5."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_zero_crossing"]


def rangayyan_zero_crossing(x, fs=1.0):
    """Zero-crossing rate.

    ``ZCR = (1/(N-1)) Σ 0.5 |sign(x[n]) - sign(x[n-1])|``.

    Parameters
    ----------
    x : array-like
    fs : float
        Sampling rate (Hz). ``zcr_per_second`` is ``zcr * fs``.

    Returns
    -------
    RichResult with keys ``zcr``, ``zcr_per_second``, ``crossings``, ``n``.

    References
    ----------
    Rangayyan Ch 5.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return with_describe_pointer(RichResult(
            title="Zero-crossing rate",
            summary_lines=[("Length", n)],
            payload={"zcr": float("nan"), "zcr_per_second": float("nan"),
                     "crossings": 0, "n": n},
        ), "rgzcr")
    s = np.sign(x); s[s == 0] = 1.0
    crossings = int(np.sum(np.abs(np.diff(s)) > 0))
    zcr = crossings / (n - 1)
    res = RichResult(
        title="Zero-crossing rate",
        summary_lines=[
            ("N", n), ("Crossings", crossings),
            ("ZCR (per sample)", zcr),
            ("ZCR (per second)", zcr * fs),
        ],
        interpretation=f"{crossings} crossings; {zcr * fs:.3g} Hz at fs={fs}.",
        payload={"zcr": float(zcr), "zcr_per_second": float(zcr * fs),
                 "crossings": crossings, "n": n},
    )
    return with_describe_pointer(res, "rgzcr")


# CANONICAL TEST
# >>> x = np.sin(2*np.pi*np.arange(100)/10.0)
# >>> r = rangayyan_zero_crossing(x, fs=100)
# >>> r["crossings"] > 0
# True


def cheatsheet():
    return "rgzcr: Zero-crossing rate — Rangayyan Ch 5"
