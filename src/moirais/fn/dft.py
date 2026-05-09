# moirais.fn — function file (hadesllm/moirais)
"""Discrete Fourier Transform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Time discovers truth. — Seneca"


def dft_compute(x, **kwargs) -> DescriptiveResult:
    """Compute the Discrete Fourier Transform of signal *x*.

    .. math::

        X(k) = \\sum_{n=0}^{N-1} x(n) \\cdot e^{-j 2\\pi k n / N}

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    W = np.exp(-2j * np.pi * k * n / N)
    X = W @ x
    return DescriptiveResult(
        name="dft_compute",
        value=float(np.max(np.abs(X))),
        extra={"spectrum": X, "N": N},
    )


dft = dft_compute


def cheatsheet() -> str:
    return "dft_compute({}) -> Discrete Fourier Transform."
