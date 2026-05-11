# morie.fn — function file (hadesllm/morie)
"""Higher-order CFA model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._containers import DescriptiveResult


def cfa_hierarchical(
    data: pd.DataFrame | np.ndarray,
    subscale_assignments: dict[str, list[str]] | None = None,
) -> DescriptiveResult:
    """Fit a higher-order CFA model.

    First-order factors load on items; a single second-order factor
    loads on the first-order factors.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response data.
    subscale_assignments : dict, optional
        {factor_name: [item_cols]}. Default: 4 MAPQ subscales.

    Returns
    -------
    DescriptiveResult
        value=dict with fit indices and loadings.

    References
    ----------
    Rindskopf, D. & Rose, T. (1988). Some theory and applications
    of confirmatory second-order factor analysis. Multivariate
    Behavioral Research, 23(1), 51-67.
    """
    if subscale_assignments is None:
        subscale_assignments = {
            "EE": [f"EE{i}" for i in range(1, 6)],
            "EA": [f"EA{i}" for i in range(1, 6)],
            "UA": [f"UA{i}" for i in range(1, 6)],
            "ER": [f"ER{i}" for i in range(1, 6)],
        }

    all_items = []
    for items in subscale_assignments.values():
        all_items.extend(items)

    if isinstance(data, pd.DataFrame):
        X = data[all_items].dropna().to_numpy(dtype=np.float64)
    else:
        X = np.asarray(data, dtype=np.float64)

    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)

    first_order = {}
    factor_scores = []
    for fname, items in subscale_assignments.items():
        idx = [all_items.index(it) for it in items]
        sub_R = R[np.ix_(idx, idx)]
        evals, evecs = np.linalg.eigh(sub_R)
        order = np.argsort(-evals)
        loads = evecs[:, order[0]] * np.sqrt(max(evals[order[0]], 0))
        first_order[fname] = {items[i]: float(loads[i]) for i in range(len(items))}
        X_sub = X[:, idx]
        X_std = (X_sub - X_sub.mean(axis=0)) / np.clip(X_sub.std(axis=0), 1e-10, None)
        factor_scores.append(X_std @ loads / max(np.sum(loads**2), 1e-10))

    F = np.column_stack(factor_scores)
    F_corr = np.corrcoef(F, rowvar=False)
    evals2, evecs2 = np.linalg.eigh(F_corr)
    order2 = np.argsort(-evals2)
    second_order = {}
    so_loads = evecs2[:, order2[0]] * np.sqrt(max(evals2[order2[0]], 0))
    for i, fname in enumerate(subscale_assignments):
        second_order[fname] = float(so_loads[i])

    var_explained = float(evals2[order2[0]] / max(evals2.sum(), 1e-10))

    return DescriptiveResult(
        name="Higher-order CFA",
        value={
            "first_order_loadings": first_order,
            "second_order_loadings": second_order,
            "variance_explained": var_explained,
        },
        extra={"n": n, "p": p, "n_factors": len(subscale_assignments)},
    )


hierarchical_cfa = cfa_hierarchical


def cheatsheet() -> str:
    return "cfa_hierarchical({}) -> Higher-order CFA model."
