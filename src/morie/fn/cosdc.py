# morie.fn -- function file (hadesllm/morie)
"""Discrete cosine transform decomposition."""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from ._containers import DescriptiveResult

_QUOTE = "He who fights with monsters should be careful lest he thereby become a monster. -- Friedrich Nietzsche"


def cosine_decompose(x, **kwargs) -> DescriptiveResult:
    """Discrete cosine transform (DCT-II) decomposition.

    Parameters
    ----------
    x : array-like
        1-D input signal.

    Returns
    -------
    DescriptiveResult
        ``value`` is signal energy; ``extra`` has ``coefficients`` (DCT),
        ``reconstruction`` (inverse DCT), ``energy_ratio`` (fraction of
        energy in top-10% coefficients).
    """
    x = np.asarray(x, dtype=float).ravel()
    coeffs = dct(x, type=2, norm="ortho")
    recon = idct(coeffs, type=2, norm="ortho")
    energy = float(np.sum(coeffs**2))
    n_top = max(1, len(coeffs) // 10)
    sorted_sq = np.sort(coeffs**2)[::-1]
    top_energy = float(np.sum(sorted_sq[:n_top]))
    ratio = top_energy / (energy + 1e-15)

    return DescriptiveResult(
        name="cosine_decompose",
        value=energy,
        extra={"coefficients": coeffs, "reconstruction": recon, "energy_ratio": ratio},
    )


cosdc = cosine_decompose


def cheatsheet() -> str:
    return "cosine_decompose({}) -> Discrete cosine transform decomposition."
