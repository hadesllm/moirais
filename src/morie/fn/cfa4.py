# morie.fn — function file (hadesllm/morie)
"""4-factor CFA using MAPQ structure (EE/EA/UA/ER)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    get_mapq_structure,
    structure_to_indices,
)


def cfa_4factor(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> dict:
    """Fit a 4-factor CFA using the MAPQ subscale structure.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses (rows = respondents, columns = items).
    items : list of str, optional
        Item column names. If None, uses MAPQ defaults (EE1-EE5, ..., ER1-ER5).

    Returns
    -------
    dict
        Keys: cfi, tli, rmsea, srmr, loadings (dict of factor -> {item: loading}),
        chi2, df, p_value.

    References
    ----------
    Hu, L. & Bentler, P.M. (1999). Cutoff criteria for fit indexes in
        covariance structure analysis. SEM, 6(1), 1-55.
    """
    structure, item_names = get_mapq_structure(items)
    S, n = cov_from_data(data, item_names)
    p = len(item_names)
    idx_struct = structure_to_indices(structure, item_names)

    result = fit_cfa(S, n, idx_struct, p)

    # Convert index-based loadings to name-based
    named_loadings = {}
    for fname, idx_map in result["loadings"].items():
        named_loadings[fname] = {item_names[i]: v for i, v in idx_map.items()}

    return {
        "cfi": result["cfi"],
        "tli": result["tli"],
        "rmsea": result["rmsea"],
        "srmr": result["srmr"],
        "loadings": named_loadings,
        "chi2": result["chi2"],
        "df": result["df"],
        "p_value": result["p_value"],
    }


def cheatsheet() -> str:
    return "cfa_4factor({}) -> 4-factor CFA using MAPQ structure (EE/EA/UA/ER)."
