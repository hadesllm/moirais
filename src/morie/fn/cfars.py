# morie.fn -- function file (hadesllm/morie)
"""Residual correlation matrix from CFA."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._cfa_engine import (
    cov_from_data,
    fit_cfa,
    structure_to_indices,
)


def cfa_residuals(
    data: pd.DataFrame | np.ndarray,
    structure: dict[str, list[str]],
    *,
    item_names: list[str] | None = None,
) -> np.ndarray:
    """Residual correlation matrix: observed minus model-implied.

    Large residuals (|r| > 0.10) suggest local misfit (Kline, 2016).

    Parameters
    ----------
    data : DataFrame or ndarray
        Item responses.
    structure : dict
        Factor name -> list of item column names.
    item_names : list of str, optional
        Ordered item names.

    Returns
    -------
    ndarray (p x p)
        Residual correlation matrix.

    References
    ----------
    Kline, R.B. (2016). Principles and Practice of Structural Equation
        Modeling. Guilford Press.
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

    # Convert to correlation residuals
    d_obs = np.sqrt(np.diag(S))
    d_obs[d_obs < 1e-10] = 1.0
    R_obs = S / np.outer(d_obs, d_obs)

    Sigma = result["implied_cov"]
    d_imp = np.sqrt(np.diag(Sigma))
    d_imp[d_imp < 1e-10] = 1.0
    R_imp = Sigma / np.outer(d_imp, d_imp)

    return R_obs - R_imp


def cheatsheet() -> str:
    return "cfa_residuals({}) -> Residual correlation matrix from CFA."
