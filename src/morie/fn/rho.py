# morie.fn — function file (hadesllm/morie)
"""Spearman rank correlation coefficient (rho)."""

from typing import Union

import numpy as np
import scipy.stats as stats
from ._richresult import RichResult


def spearman_rho(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
) -> dict:
    """
    Spearman rank correlation coefficient (rho).

    Pearson correlation applied to ranks; captures monotone (not just linear)
    associations and is robust to outliers.

    :param x: First variable (1-D array-like).
    :param y: Second variable (same length as x).
    :return: dict with keys ``rho``, ``p_value``.
    :raises ValueError: If x and y have different lengths or fewer than 3 observations.

    References
    ----------
    Spearman, C. (1904). The proof and measurement of association between two things.
        American Journal of Psychology, 15(1), 72-101.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if len(ax) != len(ay):
        raise ValueError("x and y must have the same length.")
    if len(ax) < 3:
        raise ValueError("At least 3 observations are required.")
    rho, p_val = stats.spearmanr(ax, ay)
    return RichResult(payload={"rho": float(rho), "p_value": float(p_val)})


rho = spearman_rho


def cheatsheet() -> str:
    return "spearman_rho({}) -> Spearman rank correlation coefficient (rho)."
