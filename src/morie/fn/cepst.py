# morie.fn — function file (hadesllm/morie)
"""Real cepstrum."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def real_cepstrum(
    x: np.ndarray,
    *,
    n_fft: int | None = None,
) -> SignalResult:
    """Real cepstrum: c[n] = IFFT(log|FFT(x)|).

    :param x: 1-D input signal.
    :param n_fft: FFT length (default: next power of 2 >= len(x)).
    :return: SignalResult with cepstral coefficients in ``filtered``
        and quefrency axis in ``extra["quefrency"]``.
    """
    x = np.asarray(x, dtype=float).ravel()
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(len(x))))

    X = np.fft.fft(x, n=n_fft)
    log_mag = np.log(np.abs(X) + 1e-30)
    cepstrum = np.real(np.fft.ifft(log_mag))

    return SignalResult(
        name="real_cepstrum",
        filtered=cepstrum,
        fs=0.0,
        n_samples=len(cepstrum),
        extra={"quefrency": np.arange(len(cepstrum)), "n_fft": n_fft},
    )


cepst = real_cepstrum


def cheatsheet() -> str:
    return "real_cepstrum({}) -> Real cepstrum."
