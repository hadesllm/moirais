# moirais.fn — function file (hadesllm/moirais)
"""QRS waveform feature extraction."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def qrs_waveform_features(beat: np.ndarray) -> DescriptiveResult:
    """Everything flows. — Heraclitus"""
    from moirais._waveform import qrs_waveform_features as _backend

    features = _backend(beat)
    return DescriptiveResult(
        name="qrs_waveform_features",
        value=features["amplitude"],
        extra=features,
    )


qrswv = qrs_waveform_features


def cheatsheet() -> str:
    return "qrs_waveform_features({}) -> QRS waveform feature extraction."
