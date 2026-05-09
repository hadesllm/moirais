"""Wavelet-based filtering (keep approx or detail)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Hope is like the sun."


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
    order = int(wavelet.replace("db", "").replace("haar", "1")) if wavelet != "haar" else 1
    lo = _db_filter(order)
    hi = np.array([(-1) ** k * lo[len(lo) - 1 - k] for k in range(len(lo))])
    return lo, hi


def wavelet_filter(
    x,
    wavelet: str = "db4",
    level: int = 3,
    keep: str = "approx",
) -> DescriptiveResult:
    """Wavelet-based filtering: keep approximation or detail coefficients.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name.
    level : int
        Decomposition level.
    keep : str
        'approx' to keep low-frequency, 'detail' to keep high-frequency.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wv_filters(wavelet)
    lo_r, hi_r = lo[::-1], hi[::-1]
    N = len(x)

    approxs = [x.copy()]
    details = []
    for _ in range(level):
        ca = np.convolve(approxs[-1], lo, mode="full")[::2]
        cd = np.convolve(approxs[-1], hi, mode="full")[::2]
        approxs.append(ca)
        details.append(cd)

    if keep == "approx":
        for i in range(len(details)):
            details[i] = np.zeros_like(details[i])
    else:
        approxs[-1] = np.zeros_like(approxs[-1])

    rec = approxs[-1]
    for cd in reversed(details):
        up_a = np.zeros(2 * len(rec))
        up_a[::2] = rec
        up_d = np.zeros(2 * len(cd))
        up_d[::2] = cd
        rec = np.convolve(up_a, lo_r, mode="full") + np.convolve(up_d, hi_r, mode="full")
        target = approxs.pop()
        rec = rec[: len(approxs[-1])] if approxs else rec[:N]
    filtered = rec[:N]

    return DescriptiveResult(
        name="wavelet_filter",
        value=float(np.std(filtered)),
        extra={"filtered": filtered, "wavelet": wavelet, "level": level, "keep": keep},
    )


wvflt = wavelet_filter


def cheatsheet() -> str:
    return "_db_filter({}) -> Wavelet-based filtering (keep approx or detail)."
