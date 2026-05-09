"""Wigner-Ville distribution (time-frequency representation)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def wigner_ville_fn(x: np.ndarray) -> DescriptiveResult:
    """Compute the Wigner-Ville distribution of a signal.

    Uses small signals to avoid O(n^2) memory issues.

    :param x: 1-D input signal (keep short, e.g. <= 64 samples).
    :return: DescriptiveResult with WVD matrix in extra.
    """
    from moirais._adaptive import wigner_ville

    x = np.asarray(x, dtype=float).ravel()
    wvd = wigner_ville(x)
    return DescriptiveResult(
        name="wigner_ville",
        value=None,
        extra={"wvd": wvd},
    )


wvd = wigner_ville_fn


def cheatsheet() -> str:
    return "wigner_ville_fn({}) -> Wigner-Ville distribution (time-frequency representation)."
