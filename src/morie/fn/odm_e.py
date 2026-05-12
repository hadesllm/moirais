# morie.fn -- function file (hadesllm/morie)
"""Equity metrics: representation index and disparity ratio."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def otis_demo_equity(
    group_proportions: np.ndarray,
    population_proportions: np.ndarray,
) -> ESRes:
    """Equity metrics comparing system vs population demographics.

    Parameters
    ----------
    group_proportions : ndarray
        Proportions in the system (e.g. correctional).
    population_proportions : ndarray
        Proportions in the general population.

    Returns
    -------
    ESRes
        estimate is the max disparity ratio.
    """
    gp = np.asarray(group_proportions, dtype=float)
    pp = np.asarray(population_proportions, dtype=float)
    ratios = gp / np.maximum(pp, 1e-10)
    max_disparity = float(np.max(ratios))
    ri = {f"group_{i}": float(r) for i, r in enumerate(ratios)}
    dissimilarity = 0.5 * float(np.sum(np.abs(gp - pp)))
    return ESRes(
        measure="otis_demo_equity",
        estimate=max_disparity,
        n=len(gp),
        extra={"representation_indices": ri, "dissimilarity_index": dissimilarity},
    )


odm_e = otis_demo_equity


def cheatsheet() -> str:
    return "otis_demo_equity({}) -> Equity metrics: representation index and disparity ratio."
