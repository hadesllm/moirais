"""Statistical moments of wavelet coefficients."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew

from ._containers import DescriptiveResult

_QUOTE = "Difficult to see. Always in motion is the future."


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


def wavelet_moments(
    x,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Compute statistical moments (mean, var, skew, kurtosis) of wavelet coefficients.

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
    lo, hi = _wv_filters(wavelet)

    approx = x.copy()
    all_moments = []
    labels = []
    for lv in range(1, level + 1):
        if len(approx) < len(lo):
            break
        ca = np.convolve(approx, lo, mode="full")[::2]
        cd = np.convolve(approx, hi, mode="full")[::2]
        m = {
            "mean": float(np.mean(cd)),
            "variance": float(np.var(cd)),
            "skewness": float(skew(cd)),
            "kurtosis": float(kurtosis(cd)),
        }
        all_moments.append(m)
        labels.append(f"D{lv}")
        approx = ca

    m_approx = {
        "mean": float(np.mean(approx)),
        "variance": float(np.var(approx)),
        "skewness": float(skew(approx)) if len(approx) > 2 else 0.0,
        "kurtosis": float(kurtosis(approx)) if len(approx) > 3 else 0.0,
    }
    all_moments.append(m_approx)
    labels.append(f"A{len(all_moments) - 1}")

    return DescriptiveResult(
        name="wavelet_moments",
        value=float(len(all_moments)),
        extra={"moments": all_moments, "labels": labels, "wavelet": wavelet, "level": level},
    )


wvmom = wavelet_moments


def cheatsheet() -> str:
    return "_db_filter({}) -> Statistical moments of wavelet coefficients."
