# moirais.fn — function file (hadesllm/moirais)
"""IRT-based DIF using likelihood ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def dif_irt_based(responses: np.ndarray | pd.DataFrame, group: np.ndarray | list, cdf=None, *, item_names: list[str] | None = None, alpha: float = 0.05) -> DIFResult:
    """IRT-based DIF via likelihood ratio test.

    Compares item difficulty estimates between groups using a
    simplified LR approach (comparing 1PL parameters).

    Parameters
    ----------
    responses : ndarray or DataFrame
        Binary response matrix (n x k).
    group : array-like
        Group membership (length n), two distinct values.
    item_names : list[str], optional
        Item labels.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    DIFResult
        method="IRT-LR".

    References
    ----------
    Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of
    differential item functioning using the parameters of item response
    models. In P. W. Holland & H. Wainer (Eds.), Differential Item
    Functioning. Lawrence Erlbaum.
    """
    X = np.asarray(responses, dtype=np.float64)
    g = np.asarray(group).ravel()
    n, k = X.shape
    X = np.where(np.isnan(X), 0, X)

    if item_names is None:
        item_names = list(responses.columns) if isinstance(responses, pd.DataFrame) else [f"item_{j}" for j in range(k)]

    groups = sorted(set(g))
    if len(groups) != 2:
        raise ValueError("Need exactly 2 groups.")

    rows = []
    flagged = []
    for j in range(k):
        p_ref = np.clip(X[g == groups[0], j].mean(), 0.001, 0.999)
        p_foc = np.clip(X[g == groups[1], j].mean(), 0.001, 0.999)
        n_ref = (g == groups[0]).sum()
        n_foc = (g == groups[1]).sum()
        b_ref = -np.log(p_ref / (1 - p_ref))
        b_foc = -np.log(p_foc / (1 - p_foc))

        p_all = np.clip(X[:, j].mean(), 0.001, 0.999)
        ll_compact = n_ref * (p_ref * np.log(p_all) + (1 - p_ref) * np.log(1 - p_all)) + n_foc * (
            p_foc * np.log(p_all) + (1 - p_foc) * np.log(1 - p_all)
        )
        ll_augment = n_ref * (p_ref * np.log(p_ref) + (1 - p_ref) * np.log(1 - p_ref)) + n_foc * (
            p_foc * np.log(p_foc) + (1 - p_foc) * np.log(1 - p_foc)
        )
        G2 = max(-2 * (ll_compact - ll_augment), 0)
        p_val = 1.0 - sp.chi2.cdf(G2, df=1)

        rows.append(
            {
                "item": item_names[j],
                "b_ref": float(b_ref),
                "b_focal": float(b_foc),
                "G2": float(G2),
                "p_value": float(p_val),
            }
        )
        if p_val < alpha:
            flagged.append(item_names[j])

    return DIFResult(method="IRT-LR", items=pd.DataFrame(rows), flagged=flagged)


irt_dif = dif_irt_based


def cheatsheet() -> str:
    return "dif_irt_based({}) -> IRT-based DIF using likelihood ratio."
