# morie.fn — function file (hadesllm/morie)
"""Form factor of a waveform."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def form_factor_fn(x: np.ndarray) -> DescriptiveResult:
    """Compute the form factor (RMS / mean absolute value).

    'That'You have power over your mind. — Marcus Aurelius'
    """
    from morie._waveform import form_factor as _backend

    result = _backend(x)
    return DescriptiveResult(name="form_factor", value=float(result))


alias = form_factor_fn


def cheatsheet() -> str:
    return "form_factor_fn({}) -> Form factor of a waveform."
