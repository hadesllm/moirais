# morie.fn -- function file (hadesllm/morie)
"""Generalized Likelihood Ratio change point detection."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def glr_change(x: np.ndarray, min_segment: int = 20) -> DescriptiveResult:
    """Detect change point in a signal using GLR test.

    :param x: 1-D input signal.
    :param min_segment: Minimum segment length for detection (default 20).
    :return: DescriptiveResult with split_point as value and GLR statistic in extra.
    """
    from morie._adaptive import glr_change_detect

    x = np.asarray(x, dtype=float).ravel()
    split_point, glr_val = glr_change_detect(x, min_segment=min_segment)
    return DescriptiveResult(
        name="glr_change_detect",
        value=int(split_point),
        extra={"glr": float(glr_val)},
    )


glrcd = glr_change


def cheatsheet() -> str:
    return "glr_change({}) -> Generalized Likelihood Ratio change point detection."
