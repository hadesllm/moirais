# morie.fn -- function file (hadesllm/morie)
"""Metric (weak) invariance: constrain loadings equal across groups."""

from __future__ import annotations

import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    get_mapq_structure,
    structure_to_indices,
)
from morie.fn._mapq_const import FIT_THRESHOLDS, MI_DELTA


def mi_metric(
    data: pd.DataFrame,
    group_col: str,
    structure: dict[str, list[str]] | None = None,
    *,
    items: list[str] | None = None,
    configural_fit: dict | None = None,
) -> dict:
    """Metric (weak) invariance: factor loadings constrained equal.

    Uses pooled sample covariance to approximate constrained estimation.

    Parameters
    ----------
    data : DataFrame
        Item response data with a grouping column.
    group_col : str
        Column name for the grouping variable.
    structure : dict, optional
        Factor name -> item names. Default: MAPQ 4-factor.
    items : list of str, optional
        Override item names.
    configural_fit : dict, optional
        Configural model fit dict (for delta-fit computation).

    Returns
    -------
    dict
        Keys: level, fit, delta_fit, passed.

    References
    ----------
    Chen, F.F. (2007). Sensitivity of goodness of fit indexes to lack of
        measurement invariance. SEM, 14(3), 464-504.
    """
    if structure is None:
        structure, item_names = get_mapq_structure(items)
    else:
        if items is not None:
            item_names = list(items)
        else:
            seen = []
            for v in structure.values():
                for it in v:
                    if it not in seen:
                        seen.append(it)
            item_names = seen

    p = len(item_names)
    idx_struct = structure_to_indices(structure, item_names)

    # Pooled covariance (approximation of constrained loadings model)
    S, n = cov_from_data(data, item_names)
    result = fit_cfa(S, n, idx_struct, p)

    groups = sorted(data[group_col].dropna().unique())
    n_groups = len(groups)

    # Adjust df: metric model has more constraints than configural
    # Additional constraints = (n_groups - 1) * n_loadings
    n_loadings = sum(len(v) for v in idx_struct.values())
    extra_df = (n_groups - 1) * n_loadings

    fit = {
        "chi2": result["chi2"],
        "df": result["df"] + extra_df,
        "cfi": result["cfi"],
        "tli": result["tli"],
        "rmsea": result["rmsea"],
        "srmr": result["srmr"],
        "n": n,
    }

    delta_fit = {}
    passed = fit["cfi"] >= FIT_THRESHOLDS["cfi_acceptable"]

    if configural_fit is not None:
        cfg = configural_fit.get("fit", configural_fit)
        delta_cfi = cfg.get("cfi", 1.0) - fit["cfi"]
        delta_rmsea = fit["rmsea"] - cfg.get("rmsea", 0.0)
        delta_fit = {
            "delta_cfi": float(delta_cfi),
            "delta_rmsea": float(delta_rmsea),
        }
        passed = abs(delta_cfi) <= MI_DELTA["delta_cfi"]

    return {
        "level": "metric",
        "fit": fit,
        "delta_fit": delta_fit,
        "passed": bool(passed),
    }


def cheatsheet() -> str:
    return "mi_metric({}) -> Metric (weak) invariance: constrain loadings equal across gr"
