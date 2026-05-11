# morie.fn — function file (hadesllm/morie)
"""Prediction interval for a new study from random-effects meta-analysis."""

from typing import Union

import numpy as np

from .remeta import random_effects_meta


def prediction_interval(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Prediction interval for a new study from a random-effects meta-analysis.

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like
    confidence : float, default 0.95

    Returns
    -------
    tuple[float, float]
        (lower, upper) prediction interval bounds.
    """
    result = random_effects_meta(estimates, standard_errors, confidence=confidence)
    return (
        result.extra["prediction_interval_lower"],
        result.extra["prediction_interval_upper"],
    )


pred_int = prediction_interval


def cheatsheet() -> str:
    return "prediction_interval({}) -> Prediction interval for a new study from random-effects meta"
