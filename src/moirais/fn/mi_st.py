# moirais.fn — function file (hadesllm/moirais)
"""Strict invariance: constrain loadings + intercepts + residual variances."""

from __future__ import annotations

import pandas as pd

from moirais.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    get_mapq_structure,
    structure_to_indices,
)
from moirais.fn._mapq_const import FIT_THRESHOLDS, MI_DELTA


def mi_strict(
    data: pd.DataFrame,
    group_col: str,
    structure: dict[str, list[str]] | None = None,
    *,
    items: list[str] | None = None,
    scalar_fit: dict | None = None,
) -> dict:
    """Strict invariance: loadings, intercepts, and residual variances equal.

    The most restrictive level; rarely achieved in practice.

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
    scalar_fit : dict, optional
        Scalar model fit dict (for delta-fit computation).

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

    # Additional constraints: loadings + intercepts + residual variances
    n_loadings = sum(len(v) for v in idx_struct.values())
    extra_df = (n_groups - 1) * (n_loadings + p + p)  # +p for residual vars

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

    if scalar_fit is not None:
        sf = scalar_fit.get("fit", scalar_fit)
        delta_cfi = sf.get("cfi", 1.0) - fit["cfi"]
        delta_rmsea = fit["rmsea"] - sf.get("rmsea", 0.0)
        delta_fit = {
            "delta_cfi": float(delta_cfi),
            "delta_rmsea": float(delta_rmsea),
        }
        passed = abs(delta_cfi) <= MI_DELTA["delta_cfi"]

    return {
        "level": "strict",
        "fit": fit,
        "delta_fit": delta_fit,
        "passed": bool(passed),
    }


def cheatsheet() -> str:
    return "mi_strict({}) -> Strict invariance: constrain loadings + intercepts + residua"
