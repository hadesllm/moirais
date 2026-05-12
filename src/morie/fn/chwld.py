# morie.fn -- function file (hadesllm/morie)
"""Choi-Williams distribution (reduced cross-term TFR)."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def choi_williams_fn(x: np.ndarray, sigma: float = 1.0) -> DescriptiveResult:
    """Compute the Choi-Williams distribution of a signal.

    Reduces cross-terms in the Wigner-Ville distribution.

    :param x: 1-D input signal (keep short, e.g. <= 64 samples).
    :param sigma: Kernel parameter controlling cross-term suppression (default 1.0).
    :return: DescriptiveResult with CWD matrix in extra.
    """
    from morie._adaptive import choi_williams

    x = np.asarray(x, dtype=float).ravel()
    cwd = choi_williams(x, sigma=sigma)
    return DescriptiveResult(
        name="choi_williams",
        value=None,
        extra={"cwd": cwd},
    )


chwld = choi_williams_fn


def cheatsheet() -> str:
    return "choi_williams_fn({}) -> Choi-Williams distribution (reduced cross-term TFR)."
