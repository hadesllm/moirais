"""Superposition principle verification for LTI systems."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "By all means, marry. If you get a good wife, you'll become happy; if you get a bad one, you'll become a philosopher. -- Socrates"


def superposition_test(h, x1, x2, **kwargs) -> DescriptiveResult:
    """Verify the superposition principle for an LTI system.

    Checks: h(x1 + x2) == h(x1) + h(x2)
    where h is an impulse response applied via convolution.

    Parameters
    ----------
    h : array-like
        Impulse response.
    x1 : array-like
        First input signal.
    x2 : array-like
        Second input signal.

    Returns
    -------
    DescriptiveResult
    """
    h = np.asarray(h, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y_combined = np.convolve(h, x1 + x2)
    y_sum = np.convolve(h, x1) + np.convolve(h, x2)
    error = float(np.max(np.abs(y_combined - y_sum)))
    holds = bool(error < 1e-10)
    return DescriptiveResult(
        name="superposition_test",
        value=error,
        extra={
            "max_error": error,
            "holds": holds,
            "h_length": len(h),
            "x1_length": len(x1),
            "x2_length": len(x2),
        },
    )


supps = superposition_test


def cheatsheet() -> str:
    return "superposition_test({}) -> Superposition principle verification for LTI systems."
