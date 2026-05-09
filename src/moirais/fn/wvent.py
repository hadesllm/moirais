"""Wavelet entropy from subband energy distribution."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I sense great fear in you."


def wavelet_entropy(coeffs: list) -> DescriptiveResult:
    """Wavelet entropy from the normalized energy distribution.

    .. math::

        H_w = -\\sum_j p_j \\ln p_j, \\quad p_j = E_j / E_{\\mathrm{total}}

    Parameters
    ----------
    coeffs : list
        Wavelet coefficients [cA_n, cD_n, ..., cD_1].

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``entropy``, ``relative_energies``.
    """
    energies = []
    for c in coeffs:
        c = np.asarray(c, dtype=float)
        energies.append(float(np.sum(c**2)))
    total = sum(energies)
    if total <= 0:
        return DescriptiveResult(
            name="wavelet_entropy",
            value=0.0,
            extra={"entropy": 0.0, "relative_energies": [0.0] * len(energies)},
        )
    p = np.array(energies) / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))
    return DescriptiveResult(
        name="wavelet_entropy",
        value=entropy,
        extra={"entropy": entropy, "relative_energies": (np.array(energies) / total).tolist()},
    )


wvent = wavelet_entropy


def cheatsheet() -> str:
    return "wavelet_entropy({}) -> Wavelet entropy from subband energy distribution."
