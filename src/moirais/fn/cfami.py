# moirais.fn — function file (hadesllm/moirais)
"""Modification indices for CFA models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    structure_to_indices,
)


def cfa_modindex(
    data: pd.DataFrame | np.ndarray,
    structure: dict[str, list[str]],
    *,
    item_names: list[str] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Modification indices: largest misspecifications in a CFA model.

    Approximates MI as (n-1) * residual^2 / (var_i * var_j) for cross-loadings
    and correlated residuals not in the model (Sorbom, 1989).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses.
    structure : dict
        Factor name -> list of item column names.
    item_names : list of str, optional
        Ordered item names.
    top_n : int
        Number of largest MIs to return (default 10).

    Returns
    -------
    DataFrame
        Columns: param, mi, epc (expected parameter change).

    References
    ----------
    Sorbom, D. (1989). Model modification. Psychometrika, 54(3), 371-384.
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
    residuals = result["residuals"]

    # Build set of specified paths
    specified = set()
    for fname, indices in idx_struct.items():
        for idx in indices:
            specified.add((item_names[idx], fname))

    rows = []

    # Cross-loadings: items loading on factors they don't belong to
    factor_names = list(structure.keys())
    for fi, fname in enumerate(factor_names):
        for i in range(p):
            if (item_names[i], fname) not in specified:
                # Approximate MI from residual covariance
                indices_in_factor = idx_struct[fname]
                if not indices_in_factor:
                    continue
                r_sum = sum(abs(residuals[i, j]) for j in indices_in_factor)
                mi_val = (n - 1) * (r_sum / len(indices_in_factor)) ** 2
                epc = r_sum / len(indices_in_factor)
                rows.append(
                    {
                        "param": f"{item_names[i]} =~ {fname}",
                        "mi": float(mi_val),
                        "epc": float(epc),
                    }
                )

    # Correlated residuals
    for i in range(p):
        for j in range(i + 1, p):
            # Check if i and j are in the same factor
            same = False
            for indices in idx_struct.values():
                if i in indices and j in indices:
                    same = True
                    break
            if not same:
                mi_val = (n - 1) * residuals[i, j] ** 2
                epc = residuals[i, j]
                rows.append(
                    {
                        "param": f"{item_names[i]} ~~ {item_names[j]}",
                        "mi": float(mi_val),
                        "epc": float(epc),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["param", "mi", "epc"])
    return df.nlargest(top_n, "mi").reset_index(drop=True)


def cheatsheet() -> str:
    return "cfa_modindex({}) -> Modification indices for CFA models."
