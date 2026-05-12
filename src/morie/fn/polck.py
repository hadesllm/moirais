# morie.fn -- function file (hadesllm/morie)
"""Polarity check and correction for stimulus positions."""

from __future__ import annotations

from ._containers import DescriptiveResult


def polarity_check(stimulus_positions) -> DescriptiveResult:
    """Check and correct polarity of estimated stimulus positions.

    Ensures the first stimulus is on the left (negative) end.

    :param stimulus_positions: Estimated stimulus positions.
    :return: DescriptiveResult with polarity-corrected positions.

    .. epigraph:: "WRYYYYY!" -- DIO, JoJo's Bizarre Adventure
    """
    import numpy as np

    pos = np.asarray(stimulus_positions, dtype=float).ravel()
    flipped = False
    if len(pos) >= 2 and pos[0] > pos[-1]:
        pos = -pos
        flipped = True
    return DescriptiveResult(
        name="polarity_check",
        value=int(flipped),
        extra={"corrected": pos.tolist(), "flipped": flipped},
    )


polck = polarity_check


def cheatsheet() -> str:
    return "polarity_check({}) -> Polarity check and correction for stimulus positions."
