"""Extract specific subband from wavelet decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Never tell me the odds."


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


def subband_filter(
    x,
    wavelet: str = "db4",
    level: int = 3,
    band: int = 0,
) -> DescriptiveResult:
    """Extract a specific subband from wavelet decomposition.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name.
    level : int
        Decomposition level.
    band : int
        Band index: 0 = approximation, 1..level = detail bands (1 = coarsest).

    Returns
    -------
    DescriptiveResult
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
    if band < 0 or band >= len(coeffs):
        raise ValueError(f"band must be 0..{len(coeffs) - 1}, got {band}")
    selected = coeffs[band]
    label = f"A{level}" if band == 0 else f"D{level - band + 1}"

    return DescriptiveResult(
        name="subband_filter",
        value=float(np.sum(selected**2)),
        extra={"subband": selected, "band": band, "label": label, "wavelet": wavelet, "level": level},
    )


stflt = subband_filter


def cheatsheet() -> str:
    return "_db_filter({}) -> Extract specific subband from wavelet decomposition."
