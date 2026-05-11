"""Syndrome computation for linear codes."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def syndrome_compute(H: np.ndarray, received: np.ndarray) -> DescriptiveResult:
    """Compute syndrome s = H * received mod 2.

    :param H: Parity-check matrix (r x n) over GF(2).
    :param received: Received binary vector.
    :return: DescriptiveResult with syndrome and error detection flag.
    """
    from morie.crypto._ecc import syndrome_compute as _synd

    s = _synd(np.asarray(H), np.asarray(received))
    has_error = bool(np.any(s))
    return DescriptiveResult(
        name="syndrome_compute",
        value=float(np.sum(s)),
        extra={
            "syndrome": s,
            "has_error": has_error,
            "syndrome_weight": int(np.sum(s)),
        },
    )


syndc = syndrome_compute


def cheatsheet() -> str:
    return "syndrome_compute({}) -> Syndrome computation for linear codes."
