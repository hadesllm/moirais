# moirais.fn — function file (hadesllm/moirais)
"""Pisarenko harmonic decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Already know you that which you need."


def pisarenko_fn(
    x: np.ndarray,
    nsources: int = 1,
    order: int | None = None,
    fs: float = 1.0,
) -> DescriptiveResult:
    """Pisarenko harmonic decomposition for frequency estimation.

    Uses the eigenvector corresponding to the minimum eigenvalue of
    the autocorrelation matrix to estimate sinusoidal frequencies.

    :param x: 1-D input signal.
    :param nsources: Number of sinusoidal components (default 1).
    :param order: Correlation matrix size (default nsources+1).
    :param fs: Sampling frequency (default 1.0).
    :return: DescriptiveResult with estimated frequencies.
    """
    x = np.asarray(x, dtype=float).ravel()
    if order is None:
        order = nsources + 1
    n = len(x)
    x_c = x - np.mean(x)
    R = np.zeros((order, order), dtype=complex)
    for i in range(order, n):
        seg = x_c[i - order : i][::-1]
        R += np.outer(seg, seg.conj())
    R /= n - order
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    noise_vec = eigenvectors[:, 0]
    roots = np.roots(noise_vec)
    angles = np.abs(np.angle(roots))
    frequencies = np.sort(angles * fs / (2 * np.pi))
    frequencies = frequencies[frequencies > 0][:nsources]
    return DescriptiveResult(
        name="pisarenko",
        value=None,
        extra={"frequencies": frequencies, "nsources": nsources, "fs": fs},
    )


pisrl = pisarenko_fn


def cheatsheet() -> str:
    return "pisarenko_fn({}) -> Pisarenko harmonic decomposition."
