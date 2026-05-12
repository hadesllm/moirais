# morie.fn -- function file (hadesllm/morie)
"""Matched filter detector."""

from __future__ import annotations

import numpy as np

from ._containers import SignalResult


def matched_filter_detect(x, template) -> SignalResult:
    """Apply a matched filter to detect a known template in signal *x*.

    Parameters
    ----------
    x : array-like
        Input signal to search.
    template : array-like
        Known template / reference waveform.

    Returns
    -------
    SignalResult
    """
    from morie._filters import matched_filter as _mf

    x = np.asarray(x, dtype=float)
    template = np.asarray(template, dtype=float)
    result = _mf(x, template=template)
    return SignalResult(name="matched_filter_detect", filtered=result, fs=0.0, n_samples=len(x))


mtchf = matched_filter_detect


def cheatsheet() -> str:
    return "matched_filter_detect({}) -> Matched filter detector."
