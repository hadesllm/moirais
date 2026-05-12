# morie.fn -- function file (hadesllm/morie)
"""Parseval's theorem verification."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Let the past die. Kill it, if you have to."


def parseval_verify(x, **kwargs) -> DescriptiveResult:
    r"""Verify Parseval's theorem for signal *x*.

    .. math::

        \\sum_{n=0}^{N-1} |x(n)|^2 = \\frac{1}{N} \\sum_{k=0}^{N-1} |X(k)|^2

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
        value is the ratio (should be ~1.0 if theorem holds).
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    time_energy = float(np.sum(np.abs(x) ** 2))
    X = np.fft.fft(x)
    freq_energy = float(np.sum(np.abs(X) ** 2) / N)
    ratio = freq_energy / time_energy if time_energy > 0 else float("inf")
    return DescriptiveResult(
        name="parseval_verify",
        value=ratio,
        extra={"time_energy": time_energy, "freq_energy": freq_energy, "ratio": ratio},
    )


prsv = parseval_verify


def cheatsheet() -> str:
    return "parseval_verify({}) -> Parseval's theorem verification."
