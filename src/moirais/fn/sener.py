# moirais.fn — function file (hadesllm/moirais)
"""Signal energy."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "You were the chosen one!"


def signal_energy(x, **kwargs) -> DescriptiveResult:
    """Compute the total energy of signal *x*.

    .. math::

        E = \\sum_{n=0}^{N-1} |x(n)|^2

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    energy = float(np.sum(np.abs(x) ** 2))
    return DescriptiveResult(
        name="signal_energy",
        value=energy,
        extra={"energy": energy, "n": len(x)},
    )


sener = signal_energy


def cheatsheet() -> str:
    return "signal_energy({}) -> Signal energy."
