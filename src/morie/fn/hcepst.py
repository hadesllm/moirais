# morie.fn — function file (hadesllm/morie)
"""Complex cepstrum with phase unwrapping."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def complex_cepstrum(
    x: np.ndarray,
    *,
    n_fft: int | None = None,
) -> SignalResult:
    """Complex cepstrum with phase unwrapping.

    The complex cepstrum is the inverse FFT of log(FFT(x)) where
    the log uses unwrapped phase.

    :param x: 1-D input signal.
    :param n_fft: FFT length (default: next power of 2 >= len(x)).
    :return: SignalResult with complex cepstrum in ``filtered``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(n)))

    X = np.fft.fft(x, n=n_fft)
    log_abs = np.log(np.abs(X) + 1e-30)
    phase = np.unwrap(np.angle(X))
    log_X = log_abs + 1j * phase
    cepstrum = np.real(np.fft.ifft(log_X))

    return SignalResult(
        name="complex_cepstrum",
        filtered=cepstrum,
        fs=0.0,
        n_samples=len(cepstrum),
        extra={
            "quefrency": np.arange(len(cepstrum)),
            "n_fft": n_fft,
            "original_length": n,
        },
    )


hcepst = complex_cepstrum


def cheatsheet() -> str:
    return "complex_cepstrum({}) -> Complex cepstrum with phase unwrapping."
