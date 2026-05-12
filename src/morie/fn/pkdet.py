# morie.fn -- function file (hadesllm/morie)
"""Advanced peak detection with prominence/width."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Your eyes can deceive you. Don't trust them."


def peak_detect_advanced(x, prominence=0.5, distance=10, **kwargs) -> DescriptiveResult:
    """Advanced peak detection with prominence and distance constraints.

    Parameters
    ----------
    x : array-like
        Input signal.
    prominence : float
        Minimum prominence for peaks. Default 0.5.
    distance : int
        Minimum distance between peaks. Default 10.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)

    candidates = []
    for i in range(1, n - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            candidates.append(i)

    proms = []
    for idx in candidates:
        left_min = np.min(x[max(0, idx - distance) : idx]) if idx > 0 else x[idx]
        right_min = np.min(x[idx + 1 : min(n, idx + distance + 1)]) if idx < n - 1 else x[idx]
        prom = x[idx] - max(left_min, right_min)
        proms.append(prom)

    filtered = []
    filtered_proms = []
    for i, (idx, prom) in enumerate(zip(candidates, proms)):
        if prom >= prominence:
            filtered.append(idx)
            filtered_proms.append(prom)

    if distance > 1 and len(filtered) > 1:
        keep = [True] * len(filtered)
        for i in range(len(filtered)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(filtered)):
                if not keep[j]:
                    continue
                if filtered[j] - filtered[i] < distance:
                    if x[filtered[j]] < x[filtered[i]]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        filtered = [f for f, k in zip(filtered, keep) if k]
        filtered_proms = [p for p, k in zip(filtered_proms, keep) if k]

    peaks = np.array(filtered, dtype=int)
    prominences = np.array(filtered_proms, dtype=float)

    return DescriptiveResult(
        name="peak_detect_advanced",
        value=float(len(peaks)),
        extra={
            "peaks": peaks,
            "prominences": prominences,
            "n_peaks": len(peaks),
        },
    )


pkdet = peak_detect_advanced


def cheatsheet() -> str:
    return "peak_detect_advanced({}) -> Advanced peak detection with prominence/width."
