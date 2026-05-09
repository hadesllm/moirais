# moirais.fn — function file (hadesllm/moirais)
"""Compute all fit indices for any CFA structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    structure_to_indices,
)


def cfa_fit(
    data: pd.DataFrame | np.ndarray,
    structure: dict[str, list[str]],
    *,
    item_names: list[str] | None = None,
) -> dict:
    """All fit indices for a user-specified CFA structure.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses.
    structure : dict
        Factor name -> list of item column names.
    item_names : list of str, optional
        Ordered item names. If None, inferred from structure values.

    Returns
    -------
    dict
        Keys: cfi, tli, rmsea, srmr, aic, bic, chi2, df, p_value.

    References
    ----------
    Hu, L. & Bentler, P.M. (1999). Cutoff criteria for fit indexes.
        SEM, 6(1), 1-55.
    """
    if item_names is None:
        seen = []
        for items in structure.values():
            for it in items:
                if it not in seen:
                    seen.append(it)
        item_names = seen

    S, n = cov_from_data(data, item_names)
    p = len(item_names)
    idx_struct = structure_to_indices(structure, item_names)

    result = fit_cfa(S, n, idx_struct, p)

    return {
        "cfi": result["cfi"],
        "tli": result["tli"],
        "rmsea": result["rmsea"],
        "srmr": result["srmr"],
        "aic": result["aic"],
        "bic": result["bic"],
        "chi2": result["chi2"],
        "df": result["df"],
        "p_value": result["p_value"],
    }


def cheatsheet() -> str:
    return "cfa_fit({}) -> Compute all fit indices for any CFA structure."
