# morie.fn — function file (hadesllm/morie)
"""Homomorphic deconvolution via cepstral liftering."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def homomorphic_deconvolve(
    x: np.ndarray,
    *,
    cutoff: int,
    n_fft: int | None = None,
) -> SignalResult:
    """Homomorphic deconvolution separating minimum-phase and all-pass components.

    Uses complex cepstrum + low-time liftering to extract the
    minimum-phase (impulse response) component.

    :param x: 1-D input signal (assumed to be a convolution h * e).
    :param cutoff: Liftering cutoff (quefrency index). Cepstral coefficients
        above this index are zeroed to extract the slow-varying component.
    :param n_fft: FFT length (default: next power of 2 >= len(x)).
    :return: SignalResult with minimum-phase component in ``filtered``
        and excitation in ``extra["excitation"]``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(n)))

    X = np.fft.fft(x, n=n_fft)
    log_abs = np.log(np.abs(X) + 1e-30)
    phase = np.unwrap(np.angle(X))
    log_X = log_abs + 1j * phase
    cepstrum = np.fft.ifft(log_X)

    lifter = np.zeros(n_fft)
    lifter[0] = 1.0
    c = min(cutoff, n_fft // 2)
    lifter[1:c] = 2.0
    if c < n_fft // 2:
        lifter[c] = 1.0

    cep_min = cepstrum * lifter
    log_H = np.fft.fft(cep_min)
    H = np.exp(log_H)
    h = np.real(np.fft.ifft(H))[:n]

    cep_exc = cepstrum * (1.0 - lifter)
    cep_exc[0] = cepstrum[0] - cep_min[0]
    log_E = np.fft.fft(cep_exc)
    E = np.exp(log_E)
    e = np.real(np.fft.ifft(E))[:n]

    return SignalResult(
        name="homomorphic_deconvolve",
        filtered=h,
        fs=0.0,
        n_samples=len(h),
        extra={"excitation": e, "cutoff": cutoff, "n_fft": n_fft},
    )


hdecon = homomorphic_deconvolve


def cheatsheet() -> str:
    return "homomorphic_deconvolve({}) -> Homomorphic deconvolution via cepstral liftering."
