"""Spectral kurtosis."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def spectral_kurtosis(psd: np.ndarray, freqs: np.ndarray) -> DescriptiveResult:
    """Compute the kurtosis of the power spectral density distribution.

    'Judge me by my size, do you?'
    """
    from moirais._spectral import spectral_kurtosis as _backend

    kurt = _backend(psd, freqs)
    return DescriptiveResult(
        name="spectral_kurtosis",
        value=kurt,
        extra={"kurtosis": kurt},
    )


spkrt = spectral_kurtosis


def cheatsheet() -> str:
    return "spectral_kurtosis({}) -> Spectral kurtosis."
