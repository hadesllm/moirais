"""Hamming window."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "That which does not kill us makes us stronger. -- Friedrich Nietzsche"


def hamming_window(N: int, **kwargs) -> DescriptiveResult:
    r"""Generate a Hamming window of length *N*.

    .. math::

        w(n) = 0.54 - 0.46 \\cos\\frac{2\\pi n}{N-1}

    Parameters
    ----------
    N : int
        Window length.

    Returns
    -------
    DescriptiveResult
    """
    n = np.arange(N)
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (N - 1)) if N > 1 else np.ones(1)
    return DescriptiveResult(
        name="hamming_window",
        value=float(N),
        extra={"window": w, "N": N},
    )


wnhmm = hamming_window


def cheatsheet() -> str:
    return "hamming_window({}) -> Hamming window."
