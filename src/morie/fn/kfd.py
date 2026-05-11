# morie.fn — function file (hadesllm/morie)
"""Katz fractal dimension."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def katz_fd(x: np.ndarray) -> DescriptiveResult:
    """Katz fractal dimension of a 1-D signal.

    D = log10(L) / log10(d), where L is the total path length and
    d is the diameter (max distance from first point).

    :param x: 1-D input signal.
    :return: DescriptiveResult with D in ``value``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return DescriptiveResult(name="katz_fd", value=float("nan"))

    dists = np.abs(np.diff(x))
    L = float(np.sum(dists))
    d = float(np.max(np.abs(x - x[0])))
    a = float(np.mean(dists))

    if d == 0 or a == 0:
        return DescriptiveResult(name="katz_fd", value=float("nan"))

    D = np.log10(n - 1) / (np.log10(n - 1) + np.log10(d / L))

    return DescriptiveResult(
        name="katz_fd",
        value=float(D),
        extra={"L": L, "d": d, "n": n},
    )


kfd = katz_fd


def cheatsheet() -> str:
    return "katz_fd({}) -> Katz fractal dimension."
