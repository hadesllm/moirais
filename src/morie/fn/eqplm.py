# morie.fn -- function file (hadesllm/morie)
"""Palma ratio (top 10% / bottom 40%)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def palma_ratio(
    values: np.ndarray | list[float],
) -> ESRes:
    """Compute Palma ratio: share of top 10% / share of bottom 40%.

    Parameters
    ----------
    values : array-like
        Non-negative values.

    Returns
    -------
    ESRes
    """
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) < 10:
        raise ValueError("Need at least 10 values")
    n = len(v)
    bottom40 = v[: int(n * 0.4)]
    top10 = v[int(n * 0.9) :]
    bot_sum = float(np.sum(bottom40))
    top_sum = float(np.sum(top10))
    if bot_sum <= 0:
        raise ValueError("Bottom 40% sum is zero")
    ratio = top_sum / bot_sum
    return ESRes(measure="palma_ratio", estimate=ratio, n=n, extra={"top10_share": top_sum, "bottom40_share": bot_sum})


eqplm = palma_ratio


def cheatsheet() -> str:
    return "palma_ratio({}) -> Palma ratio (top 10% / bottom 40%)."
