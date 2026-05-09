"""Signal arc length."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def signal_arc_length(x: np.ndarray) -> DescriptiveResult:
    """Compute the arc length of a discrete signal.

    'Everything flows. — Heraclitus'
    """
    from moirais._waveform import signal_arc_length as _backend

    length = _backend(x)
    return DescriptiveResult(
        name="signal_arc_length",
        value=length,
        extra={"arc_length": length},
    )


sglen = signal_arc_length


def cheatsheet() -> str:
    return "signal_arc_length({}) -> Signal arc length."
