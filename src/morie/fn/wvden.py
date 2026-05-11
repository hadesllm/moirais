"""Wavelet denoising with soft/hard thresholding."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience there is no such thing as luck."


def _daubechies_filter(order: int = 4) -> np.ndarray:
    _DB = {
        1: np.array([1.0, 1.0]) / np.sqrt(2),
        2: np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2),
        3: np.array([0.47046721, 1.14111692, 0.650365, -0.19093442, -0.12083221, 0.0498175]) / np.sqrt(2),
        4: np.array([0.32580343, 1.01094572, 0.8922014, -0.03957503, -0.26450717, 0.0436163, 0.0465036, -0.01498699])
        / np.sqrt(2),
    }
    if order not in _DB:
        raise ValueError(f"Only db1-db4 supported, got db{order}")
    return _DB[order]


def _wavelet_filters(wavelet: str = "db4"):
    if wavelet in ("haar", "db1"):
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


def _max_level(n, filt_len):
    import math

    if n < filt_len:
        return 0
    return int(math.log2(n / (filt_len - 1)))


def wavelet_denoise(
    x,
    wavelet: str = "db4",
    level: int | None = None,
    threshold: str = "soft",
) -> DescriptiveResult:
    """Wavelet denoising with soft/hard thresholding.

    Parameters
    ----------
    x : array-like
        1-D input signal.
    wavelet : str
        Wavelet name (default 'db4').
    level : int or None
        Decomposition level. Auto if None.
    threshold : str
        'soft' or 'hard' (default 'soft').

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = _wavelet_filters(wavelet)
    lo_r = lo[::-1]
    hi_r = hi[::-1]
    if level is None:
        level = max(1, _max_level(len(x), len(lo)))

    coeffs = []
    approx = x.copy()
    lengths = [len(x)]
    for _ in range(level):
        ca = np.convolve(approx, lo, mode="full")[::2]
        cd = np.convolve(approx, hi, mode="full")[::2]
        coeffs.append(cd)
        lengths.append(len(approx))
        approx = ca

    sigma = float(np.median(np.abs(coeffs[0])) / 0.6745)
    thr = sigma * np.sqrt(2 * np.log(len(x)))

    thresholded = []
    for cd in coeffs:
        if threshold == "soft":
            tc = np.sign(cd) * np.maximum(np.abs(cd) - thr, 0)
        else:
            tc = cd * (np.abs(cd) >= thr)
        thresholded.append(tc)

    rec = approx
    for cd in reversed(thresholded):
        target = lengths.pop()
        up_a = np.zeros(2 * len(rec))
        up_a[::2] = rec
        up_d = np.zeros(2 * len(cd))
        up_d[::2] = cd
        rec = np.convolve(up_a, lo_r, mode="full") + np.convolve(up_d, hi_r, mode="full")
        rec = rec[:target]

    denoised = rec[: len(x)]
    snr = 10 * np.log10(np.sum(x**2) / (np.sum((x - denoised) ** 2) + 1e-12))
    return DescriptiveResult(
        name="wavelet_denoise",
        value=float(snr),
        extra={"denoised": denoised, "threshold_value": thr, "method": threshold, "wavelet": wavelet, "level": level},
    )


wvden = wavelet_denoise


def cheatsheet() -> str:
    return "_daubechies_filter({}) -> Wavelet denoising with soft/hard thresholding."
