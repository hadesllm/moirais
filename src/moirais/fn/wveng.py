"""Wavelet energy distribution across subbands."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Impressive. Most impressive."


def wavelet_energy(coeffs: list) -> DescriptiveResult:
    """Compute energy distribution across wavelet subbands.

    .. math::

        E_j = \\sum_k |d_j(k)|^2

    Parameters
    ----------
    coeffs : list
        Wavelet coefficients [cA_n, cD_n, ..., cD_1] from ``dwt_decompose``.

    Returns
    -------
    DescriptiveResult
        ``extra`` contains ``energies`` (per subband), ``total_energy``,
        ``relative_energies`` (normalized to sum to 1).
    """
    energies = []
    for c in coeffs:
        c = np.asarray(c, dtype=float)
        energies.append(float(np.sum(c**2)))
    total = sum(energies)
    relative = [e / total if total > 0 else 0.0 for e in energies]
    return DescriptiveResult(
        name="wavelet_energy",
        value=float(total),
        extra={"energies": energies, "total_energy": total, "relative_energies": relative},
    )


wveng = wavelet_energy


def cheatsheet() -> str:
    return "wavelet_energy({}) -> Wavelet energy distribution across subbands."
