# morie.fn — function file (hadesllm/morie)
"""ESPRIT frequency estimation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def esprit_freq_fn(
    x: np.ndarray,
    nsources: int = 2,
    order: int = 16,
    fs: float = 1.0,
) -> DescriptiveResult:
    """ESPRIT (Estimation of Signal Parameters via Rotational Invariance).

    Estimates sinusoidal frequencies via the shift-invariance property
    of the signal subspace.

    :param x: 1-D input signal.
    :param nsources: Number of sinusoidal components (default 2).
    :param order: Subspace dimension (default 16).
    :param fs: Sampling frequency in Hz (default 1.0).
    :return: DescriptiveResult with estimated frequencies.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    x_c = x - np.mean(x)
    R = np.zeros((order, order), dtype=complex)
    for i in range(order, n):
        seg = x_c[i - order : i][::-1]
        R += np.outer(seg, seg.conj())
    R /= n - order
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    S = eigenvectors[:, idx[:nsources]]
    S1 = S[:-1, :]
    S2 = S[1:, :]
    Phi = np.linalg.lstsq(S1, S2, rcond=None)[0]
    eig_phi = np.linalg.eigvals(Phi)
    frequencies = np.sort(np.abs(np.angle(eig_phi) * fs / (2 * np.pi)))
    return DescriptiveResult(
        name="esprit_freq",
        value=None,
        extra={"frequencies": frequencies, "nsources": nsources, "fs": fs},
    )


esprt = esprit_freq_fn


def cheatsheet() -> str:
    return "esprit_freq_fn({}) -> ESPRIT frequency estimation."
