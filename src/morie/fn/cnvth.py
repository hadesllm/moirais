# morie.fn -- function file (hadesllm/morie)
"""Convolution theorem verification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Chewie, we're home."


def convolution_theorem_verify(x, h, **kwargs) -> DescriptiveResult:
    """Verify the convolution theorem: DFT(x*h) = DFT(x) . DFT(h).

    Both signals are zero-padded to length len(x)+len(h)-1.

    Parameters
    ----------
    x : array-like
        First signal.
    h : array-like
        Second signal (filter).

    Returns
    -------
    DescriptiveResult
        value is max absolute error between the two approaches.
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    N = len(x) + len(h) - 1
    conv_direct = np.convolve(x, h)
    X = np.fft.fft(x, n=N)
    H = np.fft.fft(h, n=N)
    conv_freq = np.real(np.fft.ifft(X * H))
    max_err = float(np.max(np.abs(conv_direct - conv_freq)))
    return DescriptiveResult(
        name="convolution_theorem_verify",
        value=max_err,
        extra={"max_error": max_err, "N_padded": N},
    )


cnvth = convolution_theorem_verify


def cheatsheet() -> str:
    return "convolution_theorem_verify({}) -> Convolution theorem verification."
