# moirais.fn — function file (hadesllm/moirais)
"""Fisher's exact test for a 2x2 contingency table."""

from typing import Union

import numpy as np
import scipy.stats as stats


def fisher_exact_test(table_2x2: Union[list, np.ndarray]) -> dict:
    """
    Fisher's exact test for a 2x2 contingency table.

    Appropriate when expected cell counts are small (< 5) and the chi-square
    approximation is unreliable.

    :param table_2x2: A 2x2 array-like [[a, b], [c, d]].
    :return: dict with keys ``odds_ratio``, ``p_value``.
    :raises ValueError: If the table is not 2x2 or contains negative values.

    References
    ----------
    Fisher, R. A. (1935). The logic of inductive inference. Journal of the Royal
        Statistical Society, 98(1), 39-82.
    """
    tbl = np.asarray(table_2x2, dtype=float)
    if tbl.shape != (2, 2):
        raise ValueError(f"table_2x2 must have shape (2, 2), got {tbl.shape}.")
    if np.any(tbl < 0):
        raise ValueError("Contingency table entries must be non-negative.")
    odds_ratio, p_val = stats.fisher_exact(tbl.astype(int))
    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_val),
        "method": "Fisher's exact test",
    }


fisher = fisher_exact_test


def cheatsheet() -> str:
    return "fisher_exact_test({}) -> Fisher's exact test for a 2x2 contingency table."
