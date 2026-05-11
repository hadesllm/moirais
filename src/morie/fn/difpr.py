# morie.fn — function file (hadesllm/morie)
"""Iterative DIF purification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from morie.fn._containers import DIFResult


def dif_purification(responses: np.ndarray | pd.DataFrame, group: np.ndarray | list, cdf=None, *, initial_anchor: list[int] | None = None, max_rounds: int = 10, alpha: float = 0.05, item_names: list[str] | None = None) -> DIFResult:
    """Iterative DIF purification procedure.

    Starts with anchor items, tests each item for DIF, removes
    flagged items from anchor, repeats until stable.

    Parameters
    ----------
    responses : ndarray or DataFrame
        Binary response matrix (n x k).
    group : array-like
        Group variable (two values).
    initial_anchor : list[int], optional
        Initial anchor item indices. Default: all items.
    max_rounds : int
        Max purification rounds (default 10).
    alpha : float
        Significance level.
    item_names : list[str], optional

    Returns
    -------
    DIFResult
        method="Purification".

    References
    ----------
    Lord, F. M. (1980). Applications of Item Response Theory to
    Practical Testing Problems. Lawrence Erlbaum.
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

    anchor = set(range(k)) if initial_anchor is None else set(initial_anchor)

    for _ in range(max_rounds):
        new_flagged = set()
        for j in range(k):
            p_ref = np.clip(X[g == groups[0], j].mean(), 0.001, 0.999)
            p_foc = np.clip(X[g == groups[1], j].mean(), 0.001, 0.999)
            n_ref = (g == groups[0]).sum()
            n_foc = (g == groups[1]).sum()
            se = np.sqrt(p_ref * (1 - p_ref) / n_ref + p_foc * (1 - p_foc) / n_foc)
            z = (p_ref - p_foc) / max(se, 1e-10)
            p_val = 2 * (1 - sp.norm.cdf(abs(z)))
            if p_val < alpha:
                new_flagged.add(j)

        new_anchor = set(range(k)) - new_flagged
        if new_anchor == anchor:
            break
        anchor = new_anchor

    flagged_names = [item_names[j] for j in sorted(set(range(k)) - anchor)]
    rows = []
    for j in range(k):
        rows.append(
            {
                "item": item_names[j],
                "anchor": j in anchor,
                "flagged": j not in anchor,
            }
        )

    return DIFResult(method="Purification", items=pd.DataFrame(rows), flagged=flagged_names)


purify_dif = dif_purification


def cheatsheet() -> str:
    return "dif_purification({}) -> Iterative DIF purification."
