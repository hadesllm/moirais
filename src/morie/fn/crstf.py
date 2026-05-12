# morie.fn -- function file (hadesllm/morie)
"""Crest factor of a waveform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def crest_factor_fn(x: np.ndarray) -> DescriptiveResult:
    """Out of chaos, comes order. -- Friedrich Nietzsche"""
    from morie._waveform import crest_factor as _backend

    result = _backend(x)
    return DescriptiveResult(name="crest_factor", value=float(result))


alias = crest_factor_fn


def cheatsheet() -> str:
    return "crest_factor_fn({}) -> Crest factor of a waveform."
