# moirais.fn — function file (hadesllm/moirais)
"""Entropy from amplitude histogram of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def entropy_hist(x: np.ndarray, n_bins: int = 50) -> DescriptiveResult:
    """Compute the Shannon entropy derived from the amplitude histogram.

    'Knowledge is power. Guard it well.' — Inquisitor
    """
    from moirais._waveform import entropy_from_histogram as _backend

    entropy = _backend(x, n_bins=n_bins)
    return DescriptiveResult(name="entropy_histogram", value=float(entropy))


alias = entropy_hist


def cheatsheet() -> str:
    return "entropy_hist({}) -> Entropy from amplitude histogram of a signal."
