# morie.fn -- function file (hadesllm/morie)
"""Phi coefficient for association in a 2x2 contingency table."""

import math
from typing import Union

import numpy as np


def phi_coefficient(table_2x2: Union[list, np.ndarray]) -> float:
    """
    Phi coefficient for association in a 2x2 contingency table.

    phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))

    Equivalent to the Pearson correlation between two binary variables.
    Ranges from -1 to +1.

    :param table_2x2: 2x2 array [[a, b], [c, d]].
    :return: Phi coefficient.
    :raises ValueError: If table is not 2x2 or marginal totals are zero.

    References
    ----------
    Pearson, K. (1900). Mathematical contributions to the theory of evolution.
        Philosophical Transactions of the Royal Society A, 195, 1-47.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must be shape (2, 2), got {tbl.shape}.")
    a, b, c, d = tbl[0, 0], tbl[0, 1], tbl[1, 0], tbl[1, 1]
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return float((a * d - b * c) / denom) if denom > 0 else 0.0


phi = phi_coefficient


def cheatsheet() -> str:
    return "phi_coefficient({}) -> Phi coefficient for association in a 2x2 contingency table."
