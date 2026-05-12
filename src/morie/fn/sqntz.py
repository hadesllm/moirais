"""Uniform signal quantization."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult

_QUOTE = "Life is really simple, but we insist on making it complicated. — Confucius"


def quantize_signal(x, bits: int = 8) -> SignalResult:
    r"""Quantize signal to b bits (uniform quantization).

    .. math::

        Q(x) = \\Delta \\cdot \\text{round}\\left(\\frac{x}{\\Delta}\\right),
        \\quad \\Delta = \\frac{x_{\\max} - x_{\\min}}{2^b - 1}

    Parameters
    ----------
    x : array-like
        Input signal.
    bits : int
        Number of quantization bits. Default 8.

    Returns
    -------
    SignalResult
    """
    x = np.asarray(x, dtype=float)
    levels = 2**bits
    xmin, xmax = np.min(x), np.max(x)
    rng = xmax - xmin
    if rng == 0:
        return SignalResult(
            name="quantize_signal",
            filtered=x.copy(),
            fs=0.0,
            n_samples=len(x),
            extra={"bits": bits, "levels": levels, "delta": 0.0},
        )
    delta = rng / (levels - 1)
    indices = np.round((x - xmin) / delta).astype(int)
    y = xmin + indices * delta
    return SignalResult(
        name="quantize_signal",
        filtered=y,
        fs=0.0,
        n_samples=len(x),
        extra={"bits": bits, "levels": levels, "delta": delta},
    )


sqntz = quantize_signal


def cheatsheet() -> str:
    return "quantize_signal({}) -> Uniform signal quantization."
