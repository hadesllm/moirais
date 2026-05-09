# moirais.fn — function file (hadesllm/moirais)
"""Point-biserial correlation between a binary and a continuous variable."""

from typing import Union

import numpy as np
import scipy.stats as stats
from ._richresult import RichResult


def point_biserial_r(
    binary_var: Union[list, np.ndarray],
    continuous_var: Union[list, np.ndarray],
) -> dict:
    """
    Point-biserial correlation between a binary and a continuous variable.

    r_pb = (M_1 - M_0) / s_total * sqrt(n_1 * n_0 / n^2)

    Equivalent to the Pearson correlation when one variable is dichotomous.

    :param binary_var: Binary variable (1-D array-like with values 0 and 1).
    :param continuous_var: Continuous variable (same length as binary_var).
    :return: dict with keys ``r``, ``p_value``.
    :raises ValueError: If inputs have different lengths or binary_var is not binary.

    References
    ----------
    Glass, G. V., & Hopkins, K. D. (1996). Statistical Methods in Education and
        Psychology (3rd ed.). Allyn & Bacon.
    """
    b = np.asarray(binary_var, dtype=float)
    c = np.asarray(continuous_var, dtype=float)
    if len(b) != len(c):
        raise ValueError("binary_var and continuous_var must have the same length.")
    unique_vals = set(np.unique(b))
    if not unique_vals.issubset({0.0, 1.0}):
        raise ValueError(f"binary_var must contain only 0 and 1, found: {unique_vals}.")
    r, p_val = stats.pointbiserialr(b, c)
    return RichResult(payload={"r": float(r), "p_value": float(p_val)})


pbr = point_biserial_r


def cheatsheet() -> str:
    return "point_biserial_r({}) -> Point-biserial correlation between a binary and a continuous"
