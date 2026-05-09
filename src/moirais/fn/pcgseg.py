# moirais.fn — function file (hadesllm/moirais)
"""S1/S2 heart sound segmentation from PCG envelope."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pcg_segment(
    envelope: np.ndarray,
    fs: float = 2000.0,
    *,
    min_gap_ms: float = 100.0,
) -> DescriptiveResult:
    """Segment a PCG envelope into S1 and S2 heart sounds.

    Thresholds the envelope, finds contiguous above-threshold
    regions, merges close peaks, and labels alternating events
    as S1 (systolic) and S2 (diastolic).

    :param envelope: 1-D Shannon-energy envelope.
    :param fs: Sampling frequency in Hz.
    :param min_gap_ms: Minimum gap between peaks in ms.
    :return: DescriptiveResult with cycle count and indices.
    """
    env = np.asarray(envelope, dtype=float).ravel()
    if len(env) < 4:
        return DescriptiveResult(
            name="pcg_segment",
            value=0,
            extra={"s1_indices": [], "s2_indices": [], "n_cycles": 0},
        )

    threshold = np.mean(env) + 0.5 * np.std(env)
    above = env > threshold
    edges = np.diff(above.astype(int))
    starts = np.where(edges == 1)[0] + 1
    stops = np.where(edges == -1)[0] + 1

    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        stops = np.concatenate([stops, [len(env)]])

    n_seg = min(len(starts), len(stops))
    if n_seg == 0:
        return DescriptiveResult(
            name="pcg_segment",
            value=0,
            extra={"s1_indices": [], "s2_indices": [], "n_cycles": 0},
        )

    starts, stops = starts[:n_seg], stops[:n_seg]

    peaks = np.array([(s + e) // 2 for s, e in zip(starts, stops)])

    min_gap = int(min_gap_ms * fs / 1000)
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] >= min_gap:
            merged.append(p)
    peaks = np.array(merged)

    s1, s2 = [], []
    for i, p in enumerate(peaks):
        if i % 2 == 0:
            s1.append(int(p))
        else:
            s2.append(int(p))

    n_cycles = min(len(s1), len(s2))

    return DescriptiveResult(
        name="pcg_segment",
        value=float(n_cycles),
        extra={
            "s1_indices": s1,
            "s2_indices": s2,
            "n_cycles": n_cycles,
            "n_peaks": len(peaks),
        },
    )


pcgseg = pcg_segment


def cheatsheet() -> str:
    return "pcg_segment({}) -> S1/S2 heart sound segmentation from PCG envelope."
