# moirais.fn — function file (hadesllm/moirais)
"""Real cepstrum."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "When I let go of what I am, I become what I might be. — Lao Tzu"


def real_cepstrum(x, **kwargs) -> DescriptiveResult:
    """Compute the real cepstrum of signal *x*.

    .. math::

        c(n) = \\text{IDFT}\\left(\\log |\\text{DFT}(x)|\\right)

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
    mag = np.abs(X)
    mag = np.where(mag == 0, np.finfo(float).tiny, mag)
    ceps = np.real(np.fft.ifft(np.log(mag)))
    return DescriptiveResult(
        name="real_cepstrum",
        value=float(ceps[0]),
        extra={"cepstrum": ceps, "N": len(x)},
    )


rceps = real_cepstrum


def cheatsheet() -> str:
    return "real_cepstrum({}) -> Real cepstrum."
