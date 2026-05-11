# morie.fn — function file (hadesllm/morie)
"""Risk score distribution overlap between groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._otis_const import DEFAULT_COLS
from ._richresult import RichResult


def risk_overlap(
    df: pd.DataFrame,
    *,
    score_col: str = DEFAULT_COLS["outcome"],
    group_col: str = DEFAULT_COLS["treatment"],
) -> dict:
    """Distribution overlap of risk scores between two groups.

    Computes the Kolmogorov-Smirnov statistic and an overlap coefficient
    (histogram intersection) between the two groups.

    Parameters
    ----------
    df : DataFrame
        Dataset with score and binary group columns.
    score_col : str
        Column with continuous risk score.
    group_col : str
        Column with binary group indicator.

    Returns
    -------
    dict
        ks_stat, ks_p, overlap_coeff, n_group0, n_group1.
    """
    tmp = df[[score_col, group_col]].dropna()
    groups = tmp[group_col].unique()
    if len(groups) < 2:
        return RichResult(payload={"ks_stat": np.nan, "ks_p": np.nan, "overlap_coeff": np.nan, "n_group0": len(tmp), "n_group1": 0})

    g0 = tmp.loc[tmp[group_col] == groups[0], score_col].values
    g1 = tmp.loc[tmp[group_col] == groups[1], score_col].values

    from scipy import stats as _st

    ks_stat, ks_p = _st.ks_2samp(g0, g1)

    # Overlap coefficient via histogram intersection
    lo = min(g0.min(), g1.min())
    hi = max(g0.max(), g1.max())
    bins = np.linspace(lo, hi, 51)
    h0, _ = np.histogram(g0, bins=bins, density=True)
    h1, _ = np.histogram(g1, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap = float(np.sum(np.minimum(h0, h1)) * bin_width)

    return {
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "overlap_coeff": overlap,
        "n_group0": len(g0),
        "n_group1": len(g1),
    }


rskov = risk_overlap


def cheatsheet() -> str:
    return "risk_overlap({}) -> Risk score distribution overlap between groups."
