# moirais.fn — function file (hadesllm/moirais)
"""Stocking-Lord DIF detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.fn._containers import DIFResult


def dif_stocking_lord(
    params_ref: dict,
    params_focal: dict,
    *,
    theta_grid: np.ndarray | None = None,
    threshold: float = 0.1,
) -> DIFResult:
    """Stocking-Lord DIF detection via characteristic curve method.

    Minimises the weighted squared difference between ICCs from
    reference and focal groups.

    Parameters
    ----------
    params_ref : dict
        {item: {"a": float, "b": float}} reference.
    params_focal : dict
        Same for focal group.
    theta_grid : ndarray, optional
        Default linspace(-4, 4, 61).
    threshold : float
        Flagging threshold for weighted area (default 0.1).

    Returns
    -------
    DIFResult
        method="Stocking-Lord".

    References
    ----------
    Stocking, M. L. & Lord, F. M. (1983). Developing a common metric
    in item response theory. Applied Psychological Measurement.
    """
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, 61)

    common = sorted(set(params_ref) & set(params_focal))
    if not common:
        raise ValueError("No common items.")

    weights = np.exp(-(theta_grid**2) / 2)
    weights /= weights.sum()

    rows = []
    flagged = []
    for item in common:
        r = params_ref[item]
        f = params_focal[item]
        a_r, b_r = r.get("a", 1.0), r["b"]
        a_f, b_f = f.get("a", 1.0), f["b"]
        P_r = 1.0 / (1.0 + np.exp(-a_r * (theta_grid - b_r)))
        P_f = 1.0 / (1.0 + np.exp(-a_f * (theta_grid - b_f)))
        wt_diff = float(np.sum(weights * (P_r - P_f) ** 2))
        rows.append({"item": item, "weighted_sq_diff": wt_diff})
        if wt_diff > threshold:
            flagged.append(item)

    return DIFResult(method="Stocking-Lord", items=pd.DataFrame(rows), flagged=flagged)


stocking_lord_dif = dif_stocking_lord


def cheatsheet() -> str:
    return "dif_stocking_lord({}) -> Stocking-Lord DIF detection."
