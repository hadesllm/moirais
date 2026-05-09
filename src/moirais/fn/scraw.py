# moirais.fn — function file (hadesllm/moirais)
"""Compute raw total/subscale scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DescriptiveResult


def raw_score(
    data: pd.DataFrame | np.ndarray,
    *,
    item_weights: np.ndarray | list | None = None,
    subscales: dict[str, list[str | int]] | None = None,
) -> DescriptiveResult:
    """Compute raw total and optional subscale scores.

    Parameters
    ----------
    data : DataFrame or ndarray
        Item response matrix.
    item_weights : array-like, optional
        Weights per item (default: equal weights of 1).
    subscales : dict, optional
        {subscale_name: [column_names_or_indices]}.

    Returns
    -------
    DescriptiveResult
        value=dict with total_score array and subscale scores.
    """
    if isinstance(data, pd.DataFrame):
        X = data.to_numpy(dtype=np.float64)
        cols = list(data.columns)
    else:
        X = np.asarray(data, dtype=np.float64)
        cols = [f"item_{j}" for j in range(X.shape[1])]

    n, k = X.shape

    if item_weights is not None:
        w = np.asarray(item_weights, dtype=np.float64)
    else:
        w = np.ones(k)

    total = X @ w

    result = {"total_score": total.tolist(), "mean": float(np.mean(total)), "sd": float(np.std(total, ddof=1))}

    if subscales is not None:
        sub_scores = {}
        for sname, sitems in subscales.items():
            if isinstance(sitems[0], str):
                idx = [cols.index(s) for s in sitems if s in cols]
            else:
                idx = list(sitems)
            sub_scores[sname] = (X[:, idx] @ w[idx]).tolist()
        result["subscales"] = sub_scores

    return DescriptiveResult(
        name="Raw scores",
        value=result,
        extra={"n": n, "k": k},
    )


raw = raw_score


def cheatsheet() -> str:
    return "raw_score({}) -> Compute raw total/subscale scores."
