# morie.fn — function file (hadesllm/morie)
"""Inverse Discrete Fourier Transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience there is no such thing as luck."


def idft_compute(X, **kwargs) -> DescriptiveResult:
    """Compute the Inverse Discrete Fourier Transform.

    .. math::

        x(n) = \\frac{1}{N} \\sum_{k=0}^{N-1} X(k) \\cdot e^{j 2\\pi k n / N}

    Parameters
    ----------
    X : array-like
        Frequency-domain signal.

    Returns
    -------
    DescriptiveResult
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)
    k = np.arange(N)
    n = k.reshape(-1, 1)
    W = np.exp(2j * np.pi * k * n / N)
    x = (W @ X) / N
    return DescriptiveResult(
        name="idft_compute",
        value=float(np.max(np.abs(x))),
        extra={"signal": x, "N": N},
    )


idft = idft_compute


def cheatsheet() -> str:
    return "idft_compute({}) -> Inverse Discrete Fourier Transform."
