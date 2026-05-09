"""Zero-crossing-based event detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Many of the truths we cling to depend on our point of view."


def zero_cross_detect(x, **kwargs) -> DescriptiveResult:
    """Detect zero-crossing events and return indices.

    Parameters
    ----------
    x : array-like
        Input signal.

    Returns
    -------
    DescriptiveResult
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return DescriptiveResult(
            name="zero_cross_detect",
            value=0.0,
            extra={"crossings": np.array([], dtype=int), "directions": np.array([])},
        )
    signs = np.sign(x)
    diff_signs = np.diff(signs)
    crossings = np.where(diff_signs != 0)[0]
    directions = np.array([1 if diff_signs[c] > 0 else -1 for c in crossings])
    rate = float(len(crossings)) / len(x) if len(x) > 0 else 0.0
    return DescriptiveResult(
        name="zero_cross_detect",
        value=float(len(crossings)),
        extra={
            "crossings": crossings,
            "directions": directions,
            "zero_crossing_rate": rate,
            "n_positive": int(np.sum(directions > 0)),
            "n_negative": int(np.sum(directions < 0)),
        },
    )


zcdet = zero_cross_detect


def cheatsheet() -> str:
    return "zero_cross_detect({}) -> Zero-crossing-based event detection."
