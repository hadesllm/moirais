# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Baseline-corrected correlation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def baseline_corrected_correlation(x: np.ndarray, y: np.ndarray) -> DescriptiveResult:
    """Compute Pearson correlation after baseline (mean) subtraction.

    'By all means, marry. If you get a good wife, you'll become happy; if you get a bad one, you'll become a philosopher. — Socrates' — Chirrut Imwe
    """
    from morie._waveform import baseline_corrected_correlation as _backend

    corr = _backend(x, y)
    return DescriptiveResult(
        name="baseline_corrected_corr",
        value=corr,
        extra={"correlation": corr},
    )


bscor = baseline_corrected_correlation


def cheatsheet() -> str:
    return "baseline_corrected_correlation({}) -> Baseline-corrected correlation."
