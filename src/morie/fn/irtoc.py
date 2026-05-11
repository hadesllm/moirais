# morie.fn — function file (hadesllm/morie)
"""IRT option characteristic curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irt_option_curves(
    data: pd.DataFrame | np.ndarray,
    theta: np.ndarray,
    *,
    item_col: str | None = None,
    n_bins: int = 10,
) -> dict:
    """Empirical option characteristic curves.

    Bins persons by theta and computes proportion selecting each option
    within each bin, for each item.

    Parameters
    ----------
    data : DataFrame or ndarray
        Items as columns, respondents as rows.
    theta : ndarray
        Person ability estimates (n,).
    item_col : str or None
        Single item to analyse. If None, analyse all items.
    n_bins : int
        Number of theta bins (default 10).

    Returns
    -------
    dict
        {item_name: DataFrame with columns theta_mid, option_1, option_2, ...}.

    References
    ----------
    Thissen, D. & Steinberg, L. (2009). Item response theory. In R. Millsap
    & A. Maydeu-Olivares (Eds.), The SAGE Handbook of Quantitative Methods.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=[f"i{i}" for i in range(data.shape[1])])

    theta = np.asarray(theta, dtype=np.float64).ravel()
    if len(theta) != len(data):
        raise ValueError(f"theta length {len(theta)} != data rows {len(data)}")

    # Bin theta
    try:
        bins = pd.qcut(theta, n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = np.zeros(len(theta), dtype=int)

    bin_mids = {}
    for b in np.unique(bins):
        mask = bins == b
        bin_mids[b] = float(np.mean(theta[mask]))

    cols = [item_col] if item_col is not None else [c for c in data.columns if np.issubdtype(data[c].dtype, np.number)]

    result = {}
    for col in cols:
        options = sorted(data[col].dropna().unique())
        rows = []
        for b in sorted(bin_mids.keys()):
            mask = bins == b
            row = {"theta_mid": bin_mids[b]}
            n_in = mask.sum()
            for opt in options:
                row[f"option_{opt}"] = float(np.mean(data.loc[mask, col] == opt)) if n_in > 0 else 0.0
            rows.append(row)
        result[col] = pd.DataFrame(rows)

    return result


def cheatsheet() -> str:
    return "irt_option_curves({}) -> IRT option characteristic curves."
