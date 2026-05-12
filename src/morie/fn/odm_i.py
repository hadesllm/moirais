# morie.fn -- function file (hadesllm/morie)
"""Diversity index (Simpson/Shannon) for demographics."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def otis_demo_index(
    counts: np.ndarray,
    *,
    method: str = "simpson",
) -> ESRes:
    """Diversity index for demographic composition.

    Parameters
    ----------
    counts : ndarray
        Counts per group.
    method : str
        'simpson' or 'shannon'.

    Returns
    -------
    ESRes
    """
    counts = np.asarray(counts, dtype=float)
    n = np.sum(counts)
    if n == 0:
        return ESRes(measure=f"demo_{method}_index", estimate=0.0, n=0)
    p = counts / n
    p = p[p > 0]
    if method == "shannon":
        idx = -float(np.sum(p * np.log(p)))
    else:
        idx = 1.0 - float(np.sum(p**2))
    return ESRes(
        measure=f"demo_{method}_index", estimate=float(idx), n=int(n), extra={"method": method, "n_groups": len(p)}
    )


odm_i = otis_demo_index


def cheatsheet() -> str:
    return "otis_demo_index({}) -> Diversity index (Simpson/Shannon) for demographics."
