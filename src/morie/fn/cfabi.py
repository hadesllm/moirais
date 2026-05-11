# morie.fn — function file (hadesllm/morie)
"""Bifactor CFA model (general + 4 specific factors)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    get_mapq_structure,
)


def cfa_bifactor(
    data: pd.DataFrame | np.ndarray,
    *,
    items: list[str] | None = None,
) -> dict:
    """Fit a bifactor CFA model: one general factor + 4 specific factors.

    Every item loads on the general factor AND its subscale-specific factor.
    Factors are constrained to be orthogonal (Reise, 2012).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses.
    items : list of str, optional
        Item column names. If None, uses MAPQ defaults.

    Returns
    -------
    dict
        Keys: cfi, tli, rmsea, srmr, loadings, chi2, df, p_value.
        loadings has keys 'General', 'EE', 'EA', 'UA', 'ER'.

    References
    ----------
    Reise, S.P. (2012). The rediscovery of bifactor measurement models.
        Multivariate Behavioral Research, 47(5), 667-696.
    """
    structure, item_names = get_mapq_structure(items)
    S, n = cov_from_data(data, item_names)
    p = len(item_names)

    # Bifactor structure: General loads on all, specific loads on subscale
    bifactor_struct = {"General": list(range(p))}
    for fname, sitems in structure.items():
        name_to_idx = {name: i for i, name in enumerate(item_names)}
        bifactor_struct[fname] = [name_to_idx[it] for it in sitems if it in name_to_idx]

    result = fit_cfa(S, n, bifactor_struct, p)

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
    return "cfa_bifactor({}) -> Bifactor CFA model (general + 4 specific factors)."
