# morie.fn — function file (hadesllm/morie)
"""Cepstral distance."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Rebellions are built on hope."


def cepstral_distance(c1, c2, **kwargs) -> DescriptiveResult:
    """Compute the Euclidean cepstral distance.

    .. math::

        d_c = \\sqrt{\\sum_{n=1}^{P} (c_1(n) - c_2(n))^2}

    Parameters
    ----------
    c1 : array-like
        Cepstral coefficient vector 1.
    c2 : array-like
        Cepstral coefficient vector 2.

    Returns
    -------
    DescriptiveResult
    """
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    minlen = min(len(c1), len(c2))
    c1 = c1[:minlen]
    c2 = c2[:minlen]
    dist = float(np.sqrt(np.sum((c1 - c2) ** 2)))
    return DescriptiveResult(
        name="cepstral_distance",
        value=dist,
        extra={"distance": dist, "length": minlen},
    )


cepds = cepstral_distance


def cheatsheet() -> str:
    return "cepstral_distance({}) -> Cepstral distance."
