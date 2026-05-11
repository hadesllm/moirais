# morie.fn — function file (hadesllm/morie)
"""Sample mean — canonical x̄ notation."""

from typing import Sequence, Union

import numpy as np


def xbar(x: Union[Sequence[float], np.ndarray]) -> float:
    """Sample arithmetic mean.

    x̄ = (1/n) Σᵢ xᵢ

    :param x: numeric sample.
    :return: arithmetic mean.

    References
    ----------
    Wooditch et al. (2021). A Beginner's Guide to Statistics for
    Criminology and Criminal Justice Using R, ch.4 (Measures of
    Central Tendency).
    """
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("cannot take mean of empty sample.")
    return float(a.mean())
