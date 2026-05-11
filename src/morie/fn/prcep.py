# morie.fn — function file (hadesllm/morie)
"""Power cepstrum."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "A long time ago in a galaxy far, far away."


def power_cepstrum(x, **kwargs) -> DescriptiveResult:
    """Compute the power cepstrum of signal *x*.

    .. math::

        c_p(n) = \\left|\\text{IDFT}\\left(\\log |\\text{DFT}(x)|^2\\right)\\right|^2

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    X = np.fft.fft(x)
    mag2 = np.abs(X) ** 2
    mag2 = np.where(mag2 == 0, np.finfo(float).tiny, mag2)
    ceps = np.abs(np.fft.ifft(np.log(mag2))) ** 2
    return DescriptiveResult(
        name="power_cepstrum",
        value=float(np.max(ceps)),
        extra={"cepstrum": ceps, "N": len(x)},
    )


prcep = power_cepstrum


def cheatsheet() -> str:
    return "power_cepstrum({}) -> Power cepstrum."
