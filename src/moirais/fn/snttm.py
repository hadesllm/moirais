"""Time served vs sentence ratio."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def sentence_time_served(
    served_days: np.ndarray,
    sentence_days: np.ndarray,
) -> ESRes:
    """Ratio of time served to sentence length.

    Parameters
    ----------
    served_days : ndarray
    sentence_days : ndarray

    Returns
    -------
    ESRes
        estimate is the mean ratio.
    """
    served = np.asarray(served_days, dtype=float)
    sentence = np.asarray(sentence_days, dtype=float)
    ratios = served / np.maximum(sentence, 1e-10)
    mean_ratio = float(np.mean(ratios))
    return ESRes(
        measure="sentence_time_served_ratio",
        estimate=mean_ratio,
        n=len(served),
        extra={"median_ratio": float(np.median(ratios)), "std": float(np.std(ratios))},
    )


snttm = sentence_time_served


def cheatsheet() -> str:
    return "sentence_time_served({}) -> Time served vs sentence ratio."
