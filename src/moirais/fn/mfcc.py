# moirais.fn — function file (hadesllm/moirais)
"""Mel-frequency cepstral coefficients."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Hope is like the sun. If you only believe it when you see it you'll never make it through the night."


def mel_cepstral_coeffs(
    x, fs: float = 16000.0, n_mfcc: int = 13, n_mels: int = 26, n_fft: int | None = None, **kwargs
) -> DescriptiveResult:
    """Compute Mel-frequency cepstral coefficients (MFCCs).

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : float
        Sampling frequency in Hz.
    n_mfcc : int
        Number of cepstral coefficients to return.
    n_mels : int
        Number of Mel filter banks.
    n_fft : int or None
        FFT size. Default: next power of 2 >= len(x).

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(len(x), 1))))
    X = np.fft.rfft(x, n=n_fft)
    power = (np.abs(X) ** 2) / n_fft

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_lo = 0.0
    mel_hi = hz_to_mel(fs / 2.0)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)

    fbank = np.zeros((n_mels, len(power)))
    for m_idx in range(n_mels):
        f_left = bins[m_idx]
        f_center = bins[m_idx + 1]
        f_right = bins[m_idx + 2]
        for k in range(f_left, f_center):
            if f_center != f_left:
                fbank[m_idx, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                fbank[m_idx, k] = (f_right - k) / (f_right - f_center)

    mel_energies = fbank @ power
    mel_energies = np.where(mel_energies == 0, np.finfo(float).tiny, mel_energies)
    log_mel = np.log(mel_energies)
    mfccs = np.zeros(n_mfcc)
    for i in range(n_mfcc):
        mfccs[i] = np.sum(log_mel * np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels))
    return DescriptiveResult(
        name="mel_cepstral_coeffs",
        value=float(mfccs[0]),
        extra={"mfcc": mfccs, "n_mfcc": n_mfcc, "n_mels": n_mels, "fs": fs},
    )


mfcc = mel_cepstral_coeffs


def cheatsheet() -> str:
    return "mel_cepstral_coeffs({}) -> Mel-frequency cepstral coefficients."
