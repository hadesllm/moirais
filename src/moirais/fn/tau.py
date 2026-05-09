"""Kendall's tau-b rank correlation coefficient."""

from typing import Union

import numpy as np
import scipy.stats as stats
from ._richresult import RichResult


def kendall_tau(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
) -> dict:
    """
    Kendall's tau-b rank correlation coefficient.

    A non-parametric measure of ordinal association that is less sensitive
    to outliers than Spearman's rho.

    :param x: First variable (1-D array-like).
    :param y: Second variable (same length as x).
    :return: dict with keys ``tau``, ``p_value``.
    :raises ValueError: If x and y have different lengths or fewer than 3 observations.

    References
    ----------
    Kendall, M. G. (1938). A new measure of rank correlation. Biometrika, 30(1-2), 81-93.
    """
    ax = np.asarray(x, dtype=float)
    ay = np.asarray(y, dtype=float)
    if len(ax) != len(ay):
        raise ValueError("x and y must have the same length.")
    if len(ax) < 3:
        raise ValueError("At least 3 observations are required.")
    tau, p_val = stats.kendalltau(ax, ay)
    return RichResult(payload={"tau": float(tau), "p_value": float(p_val)})


tau = kendall_tau


def cheatsheet() -> str:
    return "kendall_tau({}) -> Kendall's tau-b rank correlation coefficient."
