# moirais.fn — function file (hadesllm/moirais)
"""Maximal Overlap DWT (MODWT / non-decimated DWT)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Luck is what happens when preparation meets opportunity. — Seneca"


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


def maximal_overlap_dwt(
    x,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """MODWT (Maximal Overlap DWT) -- non-decimated wavelet transform.

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
    """
    x = np.asarray(x, dtype=float).ravel()
    N = len(x)
    lo, hi = _wv_filters(wavelet)

    lo_modwt = lo / np.sqrt(2)
    hi_modwt = hi / np.sqrt(2)

    approx = x.copy()
    details = []
    for j in range(level):
        lo_j = np.zeros(len(lo_modwt) * (2**j))
        hi_j = np.zeros(len(hi_modwt) * (2**j))
        for k in range(len(lo_modwt)):
            lo_j[k * (2**j)] = lo_modwt[k]
            hi_j[k * (2**j)] = hi_modwt[k]
        cd = np.convolve(approx, hi_j, mode="same")
        ca = np.convolve(approx, lo_j, mode="same")
        details.append(cd[:N])
        approx = ca[:N]

    coeffs = [approx] + list(reversed(details))
    return DescriptiveResult(
        name="maximal_overlap_dwt",
        value=float(level),
        extra={"coeffs": coeffs, "wavelet": wavelet, "level": level},
    )


maxov = maximal_overlap_dwt


def cheatsheet() -> str:
    return "_db_filter({}) -> Maximal Overlap DWT (MODWT / non-decimated DWT)."
