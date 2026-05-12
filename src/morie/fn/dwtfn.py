# morie.fn — function file (hadesllm/morie)
"""Discrete Wavelet Transform via filter bank convolution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Character is destiny. — Heraclitus"


def _daubechies_filter(order: int = 4) -> np.ndarray:
    """Return Daubechies low-pass filter coefficients for given order."""
    _DB = {
        1: np.array([1.0, 1.0]) / np.sqrt(2),
        2: np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2),
        3: np.array(
            [
                0.47046721,
                1.14111692,
                0.650365,
                -0.19093442,
                -0.12083221,
                0.0498175,
            ]
        )
        / np.sqrt(2),
        4: np.array(
            [
                0.32580343,
                1.01094572,
                0.8922014,
                -0.03957503,
                -0.26450717,
                0.0436163,
                0.0465036,
                -0.01498699,
            ]
        )
        / np.sqrt(2),
    }
    if order not in _DB:
        raise ValueError(f"Only db1-db4 supported, got db{order}")
    return _DB[order]


def _wavelet_filter(wavelet: str = "db4"):
    """Return (lo_d, hi_d) decomposition filters for named wavelet."""
    if wavelet == "haar" or wavelet == "db1":
        lo = _daubechies_filter(1)
    elif wavelet == "db2":
        lo = _daubechies_filter(2)
    elif wavelet == "db3":
        lo = _daubechies_filter(3)
    elif wavelet == "db4":
        lo = _daubechies_filter(4)
    else:
        raise ValueError(f"Unsupported wavelet '{wavelet}'; use haar/db1-db4")
    hi = np.array([(-1) ** k * lo[len(lo) - 1 - k] for k in range(len(lo))])
    return lo, hi


def _max_level(n: int, filt_len: int) -> int:
    """Maximum decomposition level for signal length n."""
    import math

    if n < filt_len:
        return 0
    return int(math.log2(n / (filt_len - 1)))


def dwt_decompose(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
) -> DescriptiveResult:
    r"""Discrete Wavelet Transform via filter bank convolution + downsampling.

    Implements the Mallat pyramidal algorithm (Rangayyan & Krishnan, Ch. 8).

    .. math::

        c_{j+1}(n) = \\sum_k h(k) \\, c_j(2n - k), \\quad
        d_{j+1}(n) = \\sum_k g(k) \\, c_j(2n - k)

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name: 'haar', 'db1', 'db2', 'db3', 'db4' (default 'db4').
    level : int or None
        Decomposition level. Auto-computed if None.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``coeffs`` (list: [cA_n, cD_n, ..., cD_1]),
        ``wavelet``, and ``level``.
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wavelet_filter(wavelet)
    if level is None:
        level = max(1, _max_level(len(x), len(lo)))
    coeffs = []
    approx = x.copy()
    for _ in range(level):
        n = len(approx)
        if n < len(lo):
            break
        ca = np.convolve(approx, lo, mode="full")[::2]
        cd = np.convolve(approx, hi, mode="full")[::2]
        coeffs.append(cd)
        approx = ca
    coeffs.append(approx)
    coeffs.reverse()
    return DescriptiveResult(
        name="dwt_decompose",
        value=float(level),
        extra={"coeffs": coeffs, "wavelet": wavelet, "level": level},
    )


dwtfn = dwt_decompose


def cheatsheet() -> str:
    return "_daubechies_filter({}) -> Discrete Wavelet Transform via filter bank convolution."
