# morie.fn -- function file (hadesllm/morie)
"""Hjorth parameters of a signal."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def hjorth_params(x: np.ndarray) -> DescriptiveResult:
    """Compute Hjorth activity, mobility, and complexity.

    'The greatest teacher, failure is.'
    """
    from morie._waveform import hjorth_parameters as _backend

    params = _backend(x)
    return DescriptiveResult(
        name="hjorth",
        value=None,
        extra=params,
    )


alias = hjorth_params


def cheatsheet() -> str:
    return "hjorth_params({}) -> Hjorth parameters of a signal."
