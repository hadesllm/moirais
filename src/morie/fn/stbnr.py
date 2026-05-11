"""Stability margin of AR model."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def stability_margin(ar_coeffs, **kwargs) -> DescriptiveResult:
    """Compute the stability margin: minimum distance of poles from unit circle.

    Parameters
    ----------
    ar_coeffs : array-like
        AR coefficients (with leading 1).

    Returns
    -------
    DescriptiveResult
    """
    ar = np.asarray(ar_coeffs, dtype=float)
    roots = np.roots(ar)
    pole_magnitudes = np.abs(roots)
    margin = float(1.0 - np.max(pole_magnitudes))
    is_stable = bool(np.all(pole_magnitudes < 1.0))
    return DescriptiveResult(
        name="stability_margin",
        value=margin,
        extra={
            "margin": margin,
            "is_stable": is_stable,
            "pole_magnitudes": pole_magnitudes,
            "poles": roots,
        },
    )


stbnr = stability_margin


def cheatsheet() -> str:
    return "stability_margin({}) -> Stability margin of AR model."
