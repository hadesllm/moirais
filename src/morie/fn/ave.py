# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Average Variance Extracted from factor loadings."""

from __future__ import annotations

import numpy as np


def ave(loads: np.ndarray) -> float:
    """Average Variance Extracted from factor loadings.

    AVE = mean(lambda^2). Values >= 0.5 indicate convergent validity
    (Fornell & Larcker, 1981).

    Parameters
    ----------
    loads : ndarray
        Standardized factor loadings (1-D array).

    Returns
    -------
    float
        AVE coefficient.
    """
    return float(np.mean(np.asarray(loads, dtype=np.float64) ** 2))


def cheatsheet() -> str:
    return "ave({}) -> Average Variance Extracted from factor loadings."
