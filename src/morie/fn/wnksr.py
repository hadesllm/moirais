"""Kaiser window."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Every generation has a legend."


def kaiser_window(N: int, beta: float = 5.0, **kwargs) -> DescriptiveResult:
    """Generate a Kaiser window of length *N* with parameter *beta*.

    Uses the modified zeroth-order Bessel function I_0.

    Parameters
    ----------
    N : int
        Window length.
    beta : float
        Shape parameter controlling sidelobe attenuation.

    Returns
    -------
    DescriptiveResult
    """
    w = np.kaiser(N, beta)
    return DescriptiveResult(
        name="kaiser_window",
        value=float(N),
        extra={"window": w, "N": N, "beta": beta},
    )


wnksr = kaiser_window


def cheatsheet() -> str:
    return "kaiser_window({}) -> Kaiser window."
