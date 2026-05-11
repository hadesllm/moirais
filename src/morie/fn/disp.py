# morie.fn — function file (hadesllm/morie)
"""Disparity index. 'Your focus determines your reality.' -- Qui-Gon Jinn"""

from __future__ import annotations

from ._containers import ESRes


def disparity_index(
    rate_group_a: float,
    rate_group_b: float,
) -> ESRes:
    """Disparity index: ratio of two group rates.

    DI = rate_a / rate_b.  A value of 1.0 indicates parity.

    Parameters
    ----------
    rate_group_a : float
        Rate for the focal group.
    rate_group_b : float
        Rate for the reference group.

    Returns
    -------
    ESRes
    """
    if rate_group_b == 0:
        raise ValueError("rate_group_b must be non-zero")
    ratio = rate_group_a / rate_group_b
    return ESRes(
        measure="Disparity index",
        estimate=float(ratio),
        extra={
            "absolute_diff": float(rate_group_a - rate_group_b),
            "rate_group_a": float(rate_group_a),
            "rate_group_b": float(rate_group_b),
        },
    )


disp = disparity_index


def cheatsheet() -> str:
    return "disparity_index({}) -> Disparity index. 'Your focus determines your reality.' -- Qu"
