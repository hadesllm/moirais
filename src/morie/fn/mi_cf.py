# morie.fn -- function file (hadesllm/morie)
"""Configural invariance: fit CFA separately per group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    get_mapq_structure,
    structure_to_indices,
)


def mi_configural(data: pd.DataFrame, group_col: str, structure: dict[str, list[str]] | None = None, cdf=None, *, items: list[str] | None = None) -> dict:
    """Configural (form) invariance: same factor structure, all params free.

    Fits the CFA model separately in each group and reports pooled fit.

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

    Returns
    -------
    dict
        Keys: level, groups, fit (pooled), group_fits, passed.

    References
    ----------
    Vandenberg, R.J. & Lance, C.E. (2000). A review and synthesis of the
        measurement invariance literature. Org. Research Methods, 3(1), 4-70.
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
    groups = sorted(data[group_col].dropna().unique())

    group_fits = {}
    total_chi2 = 0.0
    total_df = 0
    total_n = 0

    for g in groups:
        gdata = data[data[group_col] == g]
        S, n = cov_from_data(gdata, item_names)
        result = fit_cfa(S, n, idx_struct, p)
        group_fits[str(g)] = {
            "cfi": result["cfi"],
            "tli": result["tli"],
            "rmsea": result["rmsea"],
            "srmr": result["srmr"],
            "chi2": result["chi2"],
            "df": result["df"],
            "n": n,
        }
        total_chi2 += result["chi2"]
        total_df += result["df"]
        total_n += n

    from scipy import stats as sp

    pooled_p = float(1 - sp.chi2.cdf(total_chi2, total_df)) if total_df > 0 else 1.0

    from morie.fn._mapq_const import FIT_THRESHOLDS

    pooled_cfi = np.mean([f["cfi"] for f in group_fits.values()])
    passed = pooled_cfi >= FIT_THRESHOLDS["cfi_acceptable"]

    return {
        "level": "configural",
        "groups": [str(g) for g in groups],
        "fit": {
            "chi2": total_chi2,
            "df": total_df,
            "p_value": pooled_p,
            "cfi": float(pooled_cfi),
            "rmsea": float(np.mean([f["rmsea"] for f in group_fits.values()])),
            "srmr": float(np.mean([f["srmr"] for f in group_fits.values()])),
            "n": total_n,
        },
        "group_fits": group_fits,
        "passed": bool(passed),
    }


def cheatsheet() -> str:
    return "mi_configural({}) -> Configural invariance: fit CFA separately per group."
