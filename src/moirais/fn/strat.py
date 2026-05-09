"""Stratified mean estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def stratified_mean(
    data: pd.DataFrame,
    *,
    y: str = "y",
    strata: str = "stratum",
    pop_sizes: dict | None = None,
) -> DescriptiveResult:
    """Stratified population mean estimator with SE.

    Parameters
    ----------
    data : DataFrame
        Data containing outcome and stratum columns.
    y : str
        Outcome column name. Default ``"y"``.
    strata : str
        Stratum column name. Default ``"stratum"``.
    pop_sizes : dict, optional
        Population sizes per stratum. Default: proportional allocation.

    Returns
    -------
    DescriptiveResult
    """
    _validate_df(data, y, strata)
    groups = data.groupby(strata)
    strata_names = list(groups.groups.keys())
    n_h = {s: len(g) for s, g in groups}
    y_bar_h = {s: float(g[y].mean()) for s, g in groups}
    s2_h = {s: float(g[y].var(ddof=1)) for s, g in groups}
    N = sum(pop_sizes.values()) if pop_sizes else sum(n_h.values())
    W_h = {s: (pop_sizes[s] / N if pop_sizes else n_h[s] / sum(n_h.values())) for s in strata_names}
    y_bar_st = sum(W_h[s] * y_bar_h[s] for s in strata_names)
    var_st = sum(W_h[s] ** 2 * s2_h[s] / n_h[s] for s in strata_names)
    se = float(np.sqrt(var_st))
    z = stats.norm.ppf(0.975)
    return DescriptiveResult(
        name="Stratified mean",
        value=float(y_bar_st),
        extra={
            "se": se,
            "ci_lower": y_bar_st - z * se,
            "ci_upper": y_bar_st + z * se,
            "weights": W_h,
            "strata_means": y_bar_h,
            "n_strata": len(strata_names),
        },
    )


strat = stratified_mean


def cheatsheet() -> str:
    return "stratified_mean({}) -> Stratified mean estimator."
