# moirais.fn — function file (hadesllm/moirais)
"""Petrosian fractal dimension."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def petrosian_fd(x: np.ndarray) -> DescriptiveResult:
    """Petrosian fractal dimension of a 1-D signal.

    D = log10(N) / (log10(N) + log10(N / (N + 0.4 * N_delta))),
    where N_delta is the number of sign changes in the first difference.

    :param x: 1-D input signal.
    :return: DescriptiveResult with D in ``value``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 2:
        return DescriptiveResult(name="petrosian_fd", value=float("nan"))

    dx = np.diff(x)
    n_delta = int(np.sum(dx[:-1] * dx[1:] < 0))
    D = np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))

    return DescriptiveResult(
        name="petrosian_fd",
        value=float(D),
        extra={"n_delta": n_delta, "n": n},
    )


pfd = petrosian_fd


def cheatsheet() -> str:
    return "petrosian_fd({}) -> Petrosian fractal dimension."
