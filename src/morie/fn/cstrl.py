# morie.fn — function file (hadesllm/morie)
"""Release type distribution in custody."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import DescriptiveResult


def custody_release_type(
    release_types: np.ndarray,
) -> DescriptiveResult:
    """Release type distribution (parole, statutory, warrant expiry).

    Parameters
    ----------
    release_types : ndarray
        Release type labels.

    Returns
    -------
    DescriptiveResult
    """
    rt = np.asarray(release_types)
    unique, counts = np.unique(rt, return_counts=True)
    n = len(rt)
    dist = {str(u): int(c) for u, c in zip(unique, counts)}
    props = {str(u): float(c / n) for u, c in zip(unique, counts)}
    return DescriptiveResult(
        name="custody_release_type",
        value=None,
        extra={"distribution": dist, "proportions": props, "n": n},
    )


cstrl = custody_release_type


def cheatsheet() -> str:
    return "custody_release_type({}) -> Release type distribution in custody."
