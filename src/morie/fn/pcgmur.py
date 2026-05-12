# morie.fn -- function file (hadesllm/morie)
"""PCG murmur detection score."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def pcg_murmur_score(
    pcg: np.ndarray,
    fs: float,
) -> DescriptiveResult:
    """Murmur detection from a phonocardiogram signal.

    Combines band energy, spectral entropy, and Higuchi fractal
    dimension to produce a murmur likelihood score (0-1).

    :param pcg: 1-D PCG signal.
    :param fs: Sampling frequency in Hz.
    :return: DescriptiveResult with score in ``value``.
    """
    from .hfd import higuchi_fd
    from .pcgenv import pcg_envelope

    pcg = np.asarray(pcg, dtype=float).ravel()

    pcg_envelope(pcg, fs)

    from scipy.signal import butter, sosfiltfilt, welch

    sos = butter(4, [100, 400], btype="band", fs=fs, output="sos")
    hf_band = sosfiltfilt(sos, pcg)
    hf_energy = float(np.mean(hf_band**2))
    total_energy = float(np.mean(pcg**2)) + 1e-12
    hf_ratio = hf_energy / total_energy

    nperseg = min(len(pcg), 256)
    freqs, psd = welch(pcg, fs=fs, nperseg=nperseg)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))
    max_entropy = np.log(len(psd_norm))
    norm_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0

    hfd_result = higuchi_fd(pcg, kmax=10)
    fd = hfd_result.value if not np.isnan(hfd_result.value) else 1.0

    fd_score = min(1.0, max(0.0, (fd - 1.0) / 0.5))
    hf_score = min(1.0, hf_ratio * 5)
    ent_score = min(1.0, norm_entropy)
    score = 0.4 * fd_score + 0.35 * hf_score + 0.25 * ent_score

    return DescriptiveResult(
        name="pcg_murmur_score",
        value=float(score),
        extra={
            "fractal_dimension": fd,
            "hf_energy_ratio": hf_ratio,
            "spectral_entropy": norm_entropy,
            "fd_score": fd_score,
            "hf_score": hf_score,
            "ent_score": ent_score,
        },
    )


pcgmur = pcg_murmur_score


def cheatsheet() -> str:
    return "pcg_murmur_score({}) -> PCG murmur detection score."
