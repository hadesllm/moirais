# morie.fn — function file (hadesllm/morie)
"""Stigma composite score from survey responses."""

import numpy as np

from ._containers import ESRes


def stigma_index(
    responses: list | np.ndarray,
    max_per_item: int = 5,
) -> ESRes:
    """Compute stigma composite score from Likert-type items.

    Scales the raw sum to 0-100 for interpretability.

    Parameters
    ----------
    responses : array-like
        Likert-scale item responses (e.g. 1-5).
    max_per_item : int
        Maximum possible per item.

    Returns
    -------
    ESRes
    """
    a = np.asarray(responses, dtype=float)
    if len(a) == 0:
        raise ValueError("No responses provided")

    raw_sum = float(np.sum(a))
    k = len(a)
    max_possible = k * max_per_item
    scaled = raw_sum / max_possible * 100 if max_possible > 0 else 0.0

    return ESRes(
        measure="stigma_index",
        estimate=float(scaled),
        extra={"raw_sum": raw_sum, "n_items": k, "mean_item": float(np.mean(a))},
    )


mhstg = stigma_index


def cheatsheet() -> str:
    return "stigma_index({}) -> Stigma composite score from survey responses."
