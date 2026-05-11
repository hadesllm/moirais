"""Vocal tract tube model impulse response."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def vocal_tract_model_fn(
    area_function: np.ndarray,
    fs: float = 8000.0,
    n_samples: int = 1024,
) -> SignalResult:
    """Compute vocal tract impulse response from area function.

    :param area_function: Cross-sectional areas of tube sections.
    :param fs: Sampling frequency in Hz (default 8000).
    :param n_samples: Output length in samples (default 1024).
    :return: SignalResult with impulse response.
    """
    from morie._biomodel import vocal_tract_model

    area_function = np.asarray(area_function, dtype=float).ravel()
    output = vocal_tract_model(
        area_function=area_function,
        fs=fs,
        n_samples=n_samples,
    )
    n_sections = len(area_function)
    reflection = np.zeros(n_sections - 1)
    for i in range(n_sections - 1):
        s = area_function[i] + area_function[i + 1]
        if s != 0:
            reflection[i] = (area_function[i + 1] - area_function[i]) / s
    return SignalResult(
        name="vocal_tract_model",
        filtered=output,
        fs=fs,
        n_samples=len(output),
        extra={"n_sections": n_sections, "reflection_coeffs": reflection},
    )


vcltr = vocal_tract_model_fn


def cheatsheet() -> str:
    return "vocal_tract_model_fn({}) -> Vocal tract tube model impulse response."
