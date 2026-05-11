# morie.fn — function file (hadesllm/morie)
"""Synthetic control for one region."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def otis_synth_region(
    df: pd.DataFrame,
    *,
    outcome_col: str = "outcome",
    region_col: str = "region",
    period_col: str = "period",
    treated_region: str = "treated",
    pre_periods: list | None = None,
    post_periods: list | None = None,
) -> DescriptiveResult:
    """Synthetic control method for a treated region.

    Uses OLS weights on donor regions in pre-period to construct counterfactual.

    Parameters
    ----------
    df : DataFrame
    outcome_col, region_col, period_col : str
    treated_region : str
    pre_periods, post_periods : list

    Returns
    -------
    DescriptiveResult
    """
    regions = [r for r in df[region_col].unique() if r != treated_region]
    all_periods = sorted(df[period_col].unique())
    if pre_periods is None:
        mid = len(all_periods) // 2
        pre_periods = all_periods[:mid]
    if post_periods is None:
        post_periods = [p for p in all_periods if p not in pre_periods]
    pivot = df.pivot_table(index=period_col, columns=region_col, values=outcome_col)
    y_pre = pivot.loc[pivot.index.isin(pre_periods), treated_region].values
    X_pre = pivot.loc[pivot.index.isin(pre_periods), regions].values
    if X_pre.shape[0] > 0 and X_pre.shape[1] > 0:
        w = np.linalg.lstsq(X_pre, y_pre, rcond=None)[0]
        w = np.maximum(w, 0)
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum
    else:
        w = np.ones(len(regions)) / max(len(regions), 1)
    y_post = pivot.loc[pivot.index.isin(post_periods), treated_region].values
    X_post = pivot.loc[pivot.index.isin(post_periods), regions].values
    synth_post = X_post @ w if X_post.shape[0] > 0 else np.array([])
    gap = y_post - synth_post if len(synth_post) > 0 else np.array([])
    att = float(np.mean(gap)) if len(gap) > 0 else 0.0
    return DescriptiveResult(
        name="otis_synth_region",
        value=att,
        extra={"weights": {str(r): float(ww) for r, ww in zip(regions, w)}, "gap": gap.tolist(), "att": att},
    )


osyn1 = otis_synth_region


def cheatsheet() -> str:
    return "otis_synth_region({}) -> Synthetic control for one region."
