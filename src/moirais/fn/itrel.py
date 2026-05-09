# moirais.fn — function file (hadesllm/moirais)
"""Item reliability index."""

from __future__ import annotations

import numpy as np
import pandas as pd


def item_reliability_index(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Reliability index = r_it * SD_item for each item.

    The reliability index reflects an item's contribution to total-score
    reliability. Higher values indicate more reliable items.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.

    Returns
    -------
    DataFrame
        Columns: item, r_corrected, sd, reliability_index.

    References
    ----------
    Ebel, R. L. & Frisbie, D. A. (1991). Essentials of Educational
    Measurement (5th ed.). Prentice Hall.
    """
    X = np.asarray(data, dtype=np.float64)
    n, k = X.shape
    total = X.sum(axis=1)
    names = list(data.columns) if isinstance(data, pd.DataFrame) else [f"i{i}" for i in range(k)]
    rows = []
    for j in range(k):
        rest = total - X[:, j]
        sd_item = float(np.std(X[:, j], ddof=1))
        sd_rest = float(np.std(rest, ddof=1))
        if sd_item < 1e-15 or sd_rest < 1e-15:
            rc = 0.0
        else:
            rc = float(np.corrcoef(X[:, j], rest)[0, 1])
        ri = rc * sd_item
        rows.append(
            {
                "item": names[j],
                "r_corrected": rc,
                "sd": sd_item,
                "reliability_index": ri,
            }
        )
    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "item_reliability_index({}) -> Item reliability index."
