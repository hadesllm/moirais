# moirais.fn — function file (hadesllm/moirais)
"""QT interval measurement."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "There is always a bigger fish."


def qt_interval(qrs_on, t_off, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Measure QT interval from QRS onset to T-wave offset.

    Parameters
    ----------
    qrs_on : array-like of int
        QRS onset sample indices.
    t_off : array-like of int
        T-wave offset sample indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    qrs_on = np.asarray(qrs_on, dtype=int)
    t_off = np.asarray(t_off, dtype=int)
    n = min(len(qrs_on), len(t_off))
    if n == 0:
        return DescriptiveResult(
            name="qt_interval",
            value=0.0,
            extra={"qt_intervals": np.array([]), "qtc": np.array([])},
        )
    qt = (t_off[:n] - qrs_on[:n]) / fs
    rr = np.diff(qrs_on[:n]) / fs if n > 1 else np.array([1.0])
    mean_rr = float(np.mean(rr)) if len(rr) > 0 else 1.0
    qtc = qt / np.sqrt(mean_rr) if mean_rr > 0 else qt
    return DescriptiveResult(
        name="qt_interval",
        value=float(np.mean(qt)),
        extra={
            "qt_intervals": qt,
            "qtc": qtc,
            "mean_qt": float(np.mean(qt)),
            "mean_qtc": float(np.mean(qtc)),
            "fs": fs,
        },
    )


qtint = qt_interval


def cheatsheet() -> str:
    return "qt_interval({}) -> QT interval measurement."
