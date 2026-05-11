"""Mechanical resonance / Q-factor. 'Feel the vibration!' -- Snarl"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def resonance_q(
    freq: np.ndarray,
    amplitude: np.ndarray,
) -> DescriptiveResult:
    """Compute resonance frequency and quality factor (Q-factor).

    Finds the peak in a frequency response and computes Q from
    the -3dB bandwidth: Q = f_res / bandwidth.

    Parameters
    ----------
    freq : array-like
        Frequency values.
    amplitude : array-like
        Amplitude response (magnitude, not dB).

    Returns
    -------
    DescriptiveResult
        With ``value`` = Q-factor and ``extra`` containing
        resonance frequency and bandwidth.
    """
    f = np.asarray(freq, dtype=float).ravel()
    a = np.asarray(amplitude, dtype=float).ravel()
    if len(f) != len(a):
        raise ValueError("freq and amplitude must have same length")
    if len(f) < 5:
        raise ValueError("Need at least 5 frequency points")

    sorted_idx = np.argsort(f)
    f = f[sorted_idx]
    a = a[sorted_idx]

    peak_idx = np.argmax(a)
    f_res = f[peak_idx]
    a_max = a[peak_idx]

    threshold = a_max / np.sqrt(2)

    left_idx = peak_idx
    for i in range(peak_idx, -1, -1):
        if a[i] < threshold:
            left_idx = i
            break
    else:
        left_idx = 0

    right_idx = peak_idx
    for i in range(peak_idx, len(a)):
        if a[i] < threshold:
            right_idx = i
            break
    else:
        right_idx = len(a) - 1

    if left_idx < peak_idx and a[left_idx] != a[left_idx + 1]:
        f_lo = f[left_idx] + (threshold - a[left_idx]) / (a[left_idx + 1] - a[left_idx]) * (
            f[left_idx + 1] - f[left_idx]
        )
    else:
        f_lo = f[left_idx]

    if right_idx > peak_idx and a[right_idx] != a[right_idx - 1]:
        f_hi = f[right_idx - 1] + (threshold - a[right_idx - 1]) / (a[right_idx] - a[right_idx - 1]) * (
            f[right_idx] - f[right_idx - 1]
        )
    else:
        f_hi = f[right_idx]

    bandwidth = f_hi - f_lo
    Q = f_res / bandwidth if bandwidth > 0 else float("inf")

    return DescriptiveResult(
        name="resonance_q",
        value=float(Q),
        extra={
            "f_resonance": float(f_res),
            "bandwidth": float(bandwidth),
            "f_lo_3dB": float(f_lo),
            "f_hi_3dB": float(f_hi),
            "peak_amplitude": float(a_max),
        },
    )


snarl = resonance_q


def cheatsheet() -> str:
    return "resonance_q({}) -> Mechanical resonance / Q-factor. 'Feel the vibration!' -- Sn"
