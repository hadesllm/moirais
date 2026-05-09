# moirais.fn — function file (hadesllm/moirais)
"""PCG Shannon-energy envelope."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def pcg_envelope(
    pcg: np.ndarray,
    fs: float,
) -> SignalResult:
    """Shannon energy envelope for phonocardiogram signals.

    :param pcg: 1-D PCG signal.
    :param fs: Sampling frequency in Hz.
    :return: SignalResult with envelope in ``filtered``.
    """
    pcg = np.asarray(pcg, dtype=float).ravel()
    norm = pcg / (np.max(np.abs(pcg)) + 1e-12)
    shannon = -(norm**2) * np.log(norm**2 + 1e-12)

    win_len = int(0.02 * fs)
    if win_len < 1:
        win_len = 1
    kernel = np.ones(win_len) / win_len
    envelope = np.convolve(shannon, kernel, mode="same")

    return SignalResult(
        name="pcg_envelope",
        filtered=envelope,
        fs=fs,
        n_samples=len(envelope),
    )


pcgenv = pcg_envelope


def cheatsheet() -> str:
    return "pcg_envelope({}) -> PCG Shannon-energy envelope."
