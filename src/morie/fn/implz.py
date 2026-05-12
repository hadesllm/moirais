# morie.fn — function file (hadesllm/morie)
"""Estimate impulse response from input/output data."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In God we trust; all others must bring data. — W. Edwards Deming"


def impulse_from_io_fn(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    N: int = 64,
) -> DescriptiveResult:
    r"""Estimate impulse response via frequency-domain deconvolution.

    .. math::

        H(f) = \\frac{Y(f)}{X(f)}, \\quad h(n) = \\text{IFFT}\\{H(f)\\}

    :param input_signal: 1-D input signal.
    :param output_signal: 1-D output signal.
    :param N: Length of impulse response to estimate (default 64).
    :return: DescriptiveResult with estimated impulse response.
    """
    u = np.asarray(input_signal, dtype=float).ravel()
    y = np.asarray(output_signal, dtype=float).ravel()
    nfft = max(len(u), len(y), N) * 2
    U = np.fft.fft(u, n=nfft)
    Y = np.fft.fft(y, n=nfft)
    eps = np.max(np.abs(U)) * 1e-10
    H = Y / (U + eps)
    h = np.real(np.fft.ifft(H))[:N]
    return DescriptiveResult(
        name="impulse_from_io",
        value=float(N),
        extra={"impulse_response": h, "N": N},
    )


implz = impulse_from_io_fn


def cheatsheet() -> str:
    return "impulse_from_io_fn({}) -> Estimate impulse response from input/output data."
