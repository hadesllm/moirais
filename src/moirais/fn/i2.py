# moirais.fn — function file (hadesllm/moirais)
"""Higgins' I-squared heterogeneity statistic."""

from typing import Union

import numpy as np

from .remeta import random_effects_meta


def i_squared(
    estimates: Union[np.ndarray, list[float]],
    standard_errors: Union[np.ndarray, list[float]],
) -> float:
    """Compute Higgins' I-squared heterogeneity statistic.

    I^2 = max(0, (Q - (k-1))/Q) * 100

    Parameters
    ----------
    estimates : array-like
    standard_errors : array-like

    Returns
    -------
    float
        Percentage (0-100).
    """
    result = random_effects_meta(estimates, standard_errors)
    return result.extra.get("I_squared", 0.0)


isq = i_squared


def cheatsheet() -> str:
    return "i_squared({}) -> Higgins' I-squared heterogeneity statistic."
