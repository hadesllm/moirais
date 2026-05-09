# moirais.fn — function file (hadesllm/moirais)
"""Circular convolution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I don't like sand."


def circular_convolution(x, h, **kwargs) -> DescriptiveResult:
    """Compute circular convolution of *x* and *h* via DFT.

    Both sequences are treated as having period N = max(len(x), len(h)).

    Parameters
    ----------
    x : array-like
        First signal.
    h : array-like
        Second signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    N = max(len(x), len(h))
    X = np.fft.fft(x, n=N)
    H = np.fft.fft(h, n=N)
    y = np.real(np.fft.ifft(X * H))
    return DescriptiveResult(
        name="circular_convolution",
        value=float(np.max(np.abs(y))),
        extra={"output": y, "N": N},
    )


crcon = circular_convolution


def cheatsheet() -> str:
    return "circular_convolution({}) -> Circular convolution."
