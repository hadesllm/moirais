# moirais.fn — function file (hadesllm/moirais)
"""Power spectrum (periodogram)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Great, kid. Don't get cocky."


def power_spectrum(x, **kwargs) -> DescriptiveResult:
    """Compute the power spectrum |X(k)|^2 / N.

    .. math::

        P(k) = \\frac{|X(k)|^2}{N}

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    X = np.fft.fft(x)
    pwr = (np.abs(X) ** 2) / N
    return DescriptiveResult(
        name="power_spectrum",
        value=float(np.max(pwr)),
        extra={"power": pwr, "N": N},
    )


pwrsp = power_spectrum


def cheatsheet() -> str:
    return "power_spectrum({}) -> Power spectrum (periodogram)."
