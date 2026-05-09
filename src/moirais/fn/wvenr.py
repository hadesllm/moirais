"""Energy distribution across wavelet subbands."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Let the past die. Kill it, if you have to."


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


def wavelet_energy(
    x,
    wavelet: str = "db4",
    level: int = 3,
) -> DescriptiveResult:
    """Compute energy distribution across wavelet subbands.

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
    energies = []
    labels = []
    for lv in range(1, level + 1):
        ca = np.convolve(approx, lo, mode="full")[::2]
        cd = np.convolve(approx, hi, mode="full")[::2]
        e = float(np.sum(cd**2))
        energies.append(e)
        labels.append(f"D{lv}")
        approx = ca

    e_approx = float(np.sum(approx**2))
    energies.append(e_approx)
    labels.append(f"A{level}")

    total = sum(energies)
    ratios = [e / (total + 1e-12) for e in energies]

    return DescriptiveResult(
        name="wavelet_energy",
        value=float(total),
        extra={"energies": energies, "ratios": ratios, "labels": labels, "wavelet": wavelet, "level": level},
    )


wvenr = wavelet_energy


def cheatsheet() -> str:
    return "_db_filter({}) -> Energy distribution across wavelet subbands."
