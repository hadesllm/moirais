# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox positions colored by group."""

from __future__ import annotations

from ._containers import DescriptiveResult


def bb_group_color(positions, groups) -> DescriptiveResult:
    """Tag Blackbox positions with group assignments for colored plots.

    :param positions: n x d respondent position matrix.
    :param groups: Group label per respondent.
    :return: DescriptiveResult with positions and group data.

    .. epigraph:: "Gomu Gomu no..." -- Monkey D. Luffy, One Piece
    """
    import numpy as np

    X = np.asarray(positions, dtype=float)
    groups = list(groups)
    unique = sorted(set(groups))
    return DescriptiveResult(
        name="bb_group_color",
        value=len(unique),
        extra={"positions": X.tolist(), "groups": groups, "unique_groups": unique, "n": X.shape[0]},
    )


bbgrp = bb_group_color


def cheatsheet() -> str:
    return "bb_group_color({}) -> Blackbox positions colored by group."
