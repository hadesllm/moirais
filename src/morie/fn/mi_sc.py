# morie.fn -- function file (hadesllm/morie)
"""Scalar (strong) invariance: constrain loadings + intercepts."""

from __future__ import annotations

import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    get_mapq_structure,
    structure_to_indices,
)
from morie.fn._mapq_const import FIT_THRESHOLDS, MI_DELTA


def mi_scalar(
    data: pd.DataFrame,
    group_col: str,
    structure: dict[str, list[str]] | None = None,
    *,
    items: list[str] | None = None,
    metric_fit: dict | None = None,
) -> dict:
    """Scalar (strong) invariance: loadings and intercepts constrained equal.

    Required for meaningful comparison of latent means across groups.

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
    metric_fit : dict, optional
        Metric model fit dict (for delta-fit computation).

    Returns
    -------
    dict
        Keys: level, fit, delta_fit, passed.

    References
    ----------
    Meredith, W. (1993). Measurement invariance, factor analysis, and
        factorial invariance. Psychometrika, 58(4), 525-543.
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

    S, n = cov_from_data(data, item_names)
    result = fit_cfa(S, n, idx_struct, p)

    groups = sorted(data[group_col].dropna().unique())
    n_groups = len(groups)

    # Additional constraints: loadings + intercepts
    n_loadings = sum(len(v) for v in idx_struct.values())
    extra_df = (n_groups - 1) * (n_loadings + p)  # intercepts for all items

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

    if metric_fit is not None:
        mf = metric_fit.get("fit", metric_fit)
        delta_cfi = mf.get("cfi", 1.0) - fit["cfi"]
        delta_rmsea = fit["rmsea"] - mf.get("rmsea", 0.0)
        delta_fit = {
            "delta_cfi": float(delta_cfi),
            "delta_rmsea": float(delta_rmsea),
        }
        passed = abs(delta_cfi) <= MI_DELTA["delta_cfi"]

    return {
        "level": "scalar",
        "fit": fit,
        "delta_fit": delta_fit,
        "passed": bool(passed),
    }


def cheatsheet() -> str:
    return "mi_scalar({}) -> Scalar (strong) invariance: constrain loadings + intercepts."
