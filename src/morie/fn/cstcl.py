# morie.fn -- function file (hadesllm/morie)
"""Custody classification level distribution and transitions."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def custody_classification(
    levels: np.ndarray,
) -> DescriptiveResult:
    """Classification level distribution.

    Parameters
    ----------
    levels : ndarray
        Security/classification levels.

    Returns
    -------
    DescriptiveResult
    """
    levels = np.asarray(levels)
    unique, counts = np.unique(levels, return_counts=True)
    n = len(levels)
    dist = {str(u): int(c) for u, c in zip(unique, counts)}
    props = {str(u): float(c / n) for u, c in zip(unique, counts)}
    return DescriptiveResult(
        name="custody_classification",
        value=None,
        extra={"distribution": dist, "proportions": props, "n": n, "n_levels": len(unique)},
    )


cstcl = custody_classification


def cheatsheet() -> str:
    return "custody_classification({}) -> Custody classification level distribution and transitions."
