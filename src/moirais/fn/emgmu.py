# moirais.fn — function file (hadesllm/moirais)
"""EMG motor unit action potential detection.

Reference: Rangayyan, R.M. & Krishnan, S. (2024). *Biomedical Signal
Analysis*, 3rd ed. IEEE/Wiley, Chapter 14.
"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

__all__ = ['emgmu']

_QUOTE = "The unexamined statistic is not worth reporting. — adapted from Socrates"


def emgmu(
    x: np.ndarray,
    fs: float = 1000.0,
    *,
    threshold_factor: float = 3.0,
    min_duration_ms: float = 2.0,
    max_duration_ms: float = 30.0,
    refractory_ms: float = 5.0,
) -> DescriptiveResult:
    """Detect motor unit action potentials (MUAPs).

    Parameters
    ----------
    x : array-like
        1-D EMG signal.
    fs : float
        Sampling frequency in Hz.
    threshold_factor : float
        Detection threshold as multiple of RMS amplitude.
    min_duration_ms : float
        Minimum MUAP duration (ms).
    max_duration_ms : float
        Maximum MUAP duration (ms).
    refractory_ms : float
        Refractory period between detections (ms).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    rms = np.sqrt(np.mean(x ** 2))
    threshold = threshold_factor * rms

    min_samples = max(1, int(min_duration_ms * fs / 1000))
    max_samples = max(1, int(max_duration_ms * fs / 1000))
    refr_samples = max(1, int(refractory_ms * fs / 1000))

    rect = np.abs(x)
    muaps = []
    i = 0
    while i < n:
        if rect[i] > threshold:
            start = i
            while i < n and rect[i] > threshold:
                i += 1
            dur = i - start
            if min_samples <= dur <= max_samples:
                peak_idx = start + np.argmax(rect[start:i])
                muaps.append({
                    "onset": start,
                    "offset": i,
                    "peak_idx": int(peak_idx),
                    "peak_amp": float(x[peak_idx]),
                    "duration_ms": dur / fs * 1000,
                })
                i = max(i, start + refr_samples)
        else:
            i += 1

    durations = [m["duration_ms"] for m in muaps]
    amps = [abs(m["peak_amp"]) for m in muaps]

    return DescriptiveResult(
        name="emgmu",
        value=float(len(muaps)),
        extra={
            "muaps": muaps,
            "n_muaps": len(muaps),
            "mean_duration_ms": float(np.mean(durations)) if durations else 0.0,
            "mean_amplitude": float(np.mean(amps)) if amps else 0.0,
            "firing_rate_hz": len(muaps) / (n / fs) if n > 0 else 0.0,
        },
    )


def cheatsheet() -> str:
    return "emgmu({}) -> EMG MUAP detection."
