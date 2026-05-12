# morie.fn -- function file (hadesllm/morie)
"""Multiresolution analysis (MRA) decomposition/reconstruction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The greatest teacher, failure is."


def _db_filter(order=4):
    _DB = {
        1: np.array([1.0, 1.0]) / np.sqrt(2),
        2: np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2),
        3: np.array([0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]) / np.sqrt(2),
        4: np.array([0.32580343, 1.01094572, 0.8922014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699])
        / np.sqrt(2),
    }
    return _DB.get(order, _DB[4])


def _wv_filters(wavelet="db4"):
    order = int(wavelet.replace("db", "").replace("haar", "1")) if wavelet not in ("haar",) else 1
    lo = _db_filter(order)
    hi = np.array([(-1) ** k * lo[len(lo) - 1 - k] for k in range(len(lo))])
    return lo, hi


def multiresolution_analysis(
    x,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Full multiresolution analysis: decompose and reconstruct each level.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int
        Decomposition level.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wv_filters(wavelet)
    lo_r, hi_r = lo[::-1], hi[::-1]
    N = len(x)

    approxs = [x.copy()]
    details_coeffs = []
    for _ in range(level):
        ca = np.convolve(approxs[-1], lo, mode="full")[::2]
        cd = np.convolve(approxs[-1], hi, mode="full")[::2]
        approxs.append(ca)
        details_coeffs.append(cd)

    detail_signals = []
    for lv in range(level):
        cd = details_coeffs[lv]
        rec = np.zeros(2 * len(cd))
        rec[::2] = cd
        for back in range(lv, -1, -1):
            rec = np.convolve(rec, hi_r if back == lv else lo_r, mode="full")
            target_len = len(approxs[back])
            rec = rec[:target_len]
        detail_signals.append(rec[:N])

    approx_signal = approxs[-1]
    rec = np.zeros(2 * len(approx_signal))
    rec[::2] = approx_signal
    for back in range(level - 1, -1, -1):
        rec = np.convolve(rec, lo_r, mode="full")
        rec = rec[: len(approxs[back])]
    approx_signal_full = rec[:N]

    return DescriptiveResult(
        name="multiresolution_analysis",
        value=float(level),
        extra={
            "approximation": approx_signal_full,
            "details": detail_signals,
            "level": level,
            "wavelet": wavelet,
        },
    )


mltan = multiresolution_analysis


def cheatsheet() -> str:
    return "_db_filter({}) -> Multiresolution analysis (MRA) decomposition/reconstruction."
