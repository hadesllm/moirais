# morie.fn — function file (hadesllm/morie)
"""Matched filter bank detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "In my experience, there is no such thing as luck."


def matched_filter_bank(signal, templates, fs: float = 1.0, **kwargs) -> DescriptiveResult:
    """Apply multiple matched filter templates and return best match.

    Parameters
    ----------
    signal : array-like
        Input signal.
    templates : list of array-like
        Template waveforms to match against.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    DescriptiveResult
    """
    signal = np.asarray(signal, dtype=float)
    correlations = []
    for tmpl in templates:
        tmpl = np.asarray(tmpl, dtype=float)
        if len(tmpl) == 0:
            correlations.append(0.0)
            continue
        norm_s = np.linalg.norm(signal)
        norm_t = np.linalg.norm(tmpl)
        if norm_s == 0 or norm_t == 0:
            correlations.append(0.0)
            continue
        corr = np.correlate(signal, tmpl, mode="full")
        correlations.append(float(np.max(np.abs(corr)) / (norm_s * norm_t)))
    correlations = np.array(correlations)
    best_idx = int(np.argmax(correlations))
    return DescriptiveResult(
        name="matched_filter_bank",
        value=float(correlations[best_idx]),
        extra={
            "best_template_index": best_idx,
            "correlations": correlations,
            "n_templates": len(templates),
            "fs": fs,
        },
    )


mtchb = matched_filter_bank


def cheatsheet() -> str:
    return "matched_filter_bank({}) -> Matched filter bank detection."
