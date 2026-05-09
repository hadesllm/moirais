"""Multi-level wavelet decomposition returning all coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The more you know, the more you realize you don't know. — Aristotle"


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


def wavelet_decompose(
    x,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Multi-level wavelet decomposition.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name.
    level : int
        Decomposition level.

    Returns
    -------
    DescriptiveResult
        ``extra['coeffs']`` = [cA_n, cD_n, ..., cD_1].
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wv_filters(wavelet)

    approx = x.copy()
    details = []
    for _ in range(level):
        if len(approx) < len(lo):
            break
        ca = np.convolve(approx, lo, mode="full")[::2]
        cd = np.convolve(approx, hi, mode="full")[::2]
        details.append(cd)
        approx = ca

    coeffs = [approx] + list(reversed(details))
    actual_level = len(details)

    return DescriptiveResult(
        name="wavelet_decompose",
        value=float(actual_level),
        extra={"coeffs": coeffs, "wavelet": wavelet, "level": actual_level},
    )


wvdec = wavelet_decompose


def cheatsheet() -> str:
    return "_db_filter({}) -> Multi-level wavelet decomposition returning all coefficients"
