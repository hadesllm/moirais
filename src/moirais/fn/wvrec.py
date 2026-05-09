"""Reconstruct signal from wavelet coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Strike me down, and I will become more powerful."


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


def wavelet_reconstruct(coeffs, wavelet: str = "db4") -> DescriptiveResult:
    """Reconstruct signal from wavelet coefficients.

    Parameters
    ----------
    coeffs : list of array-like
        [cA_n, cD_n, ..., cD_1] as returned by wavelet_decompose.
    wavelet : str
        Wavelet name.

    Returns
    -------
    DescriptiveResult
    """
    lo, hi = _wv_filters(wavelet)
    lo_r = lo[::-1]
    hi_r = hi[::-1]

    approx = np.asarray(coeffs[0], dtype=float)
    for i in range(1, len(coeffs)):
        cd = np.asarray(coeffs[i], dtype=float)
        n = max(len(approx), len(cd))
        up_a = np.zeros(2 * n)
        up_a[: 2 * len(approx) : 2] = approx
        up_d = np.zeros(2 * n)
        up_d[: 2 * len(cd) : 2] = cd
        ra = np.convolve(up_a, lo_r, mode="full")
        rd = np.convolve(up_d, hi_r, mode="full")
        m = min(len(ra), len(rd))
        approx = ra[:m] + rd[:m]

    return DescriptiveResult(
        name="wavelet_reconstruct",
        value=float(len(approx)),
        extra={"reconstructed": approx, "wavelet": wavelet, "n_levels": len(coeffs) - 1},
    )


wvrec = wavelet_reconstruct


def cheatsheet() -> str:
    return "_db_filter({}) -> Reconstruct signal from wavelet coefficients."
