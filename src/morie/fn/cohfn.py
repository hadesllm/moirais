# morie.fn — function file (hadesllm/morie)
"""Magnitude-squared coherence function."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I have spoken."


def coherence_function_fn(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> DescriptiveResult:
    r"""Compute magnitude-squared coherence between two signals.

    .. math::

        \\gamma^2(f) = \\frac{|S_{xy}(f)|^2}{S_{xx}(f) S_{yy}(f)}

    Uses Welch's method with 50% overlapping segments.

    :param x: 1-D input signal.
    :param y: 1-D input signal (same length as x).
    :param fs: Sampling frequency (default 1.0).
    :param nperseg: Segment length for Welch averaging (default 256).
    :return: DescriptiveResult with frequency vector and coherence.
    """
    from scipy.signal import coherence

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    nperseg = min(nperseg, n)
    freqs, coh = coherence(x[:n], y[:n], fs=fs, nperseg=nperseg)
    mean_coh = float(np.mean(coh))
    return DescriptiveResult(
        name="coherence_function",
        value=mean_coh,
        extra={"frequencies": freqs, "coherence": coh, "fs": fs, "nperseg": nperseg},
    )


cohfn = coherence_function_fn


def cheatsheet() -> str:
    return "coherence_function_fn({}) -> Magnitude-squared coherence function."
