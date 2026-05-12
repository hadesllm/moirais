# morie.fn -- function file (hadesllm/morie)
"""Standardized factor loadings from CFA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    structure_to_indices,
)


def cfa_loadings(
    data: pd.DataFrame | np.ndarray,
    structure: dict[str, list[str]],
    *,
    item_names: list[str] | None = None,
) -> pd.DataFrame:
    """Standardized factor loadings for a CFA model.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses.
    structure : dict
        Factor name -> list of item column names.
    item_names : list of str, optional
        Ordered item names. If None, inferred from structure.

    Returns
    -------
    DataFrame
        Columns: item, factor, loading.

    References
    ----------
    Brown, T.A. (2015). Confirmatory Factor Analysis for Applied Research.
        Guilford Press.
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

    # Standardize loadings: divide by sqrt(implied variance)
    lam = result["lambda"]
    implied_var = np.diag(result["implied_cov"])
    sd = np.sqrt(np.maximum(implied_var, 1e-10))

    rows = []
    for fi, fname in enumerate(structure):
        for idx in idx_struct[fname]:
            std_loading = lam[idx, fi] / sd[idx] if sd[idx] > 1e-10 else 0.0
            rows.append(
                {
                    "item": item_names[idx],
                    "factor": fname,
                    "loading": float(std_loading),
                }
            )

    return pd.DataFrame(rows)


def cheatsheet() -> str:
    return "cfa_loadings({}) -> Standardized factor loadings from CFA."
