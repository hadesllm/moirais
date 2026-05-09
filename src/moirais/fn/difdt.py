# moirais.fn — function file (hadesllm/moirais)
"""Delta plot method for DIF detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._containers import DIFResult


def dif_delta_plot(
    p_values_ref: np.ndarray | list,
    p_values_focal: np.ndarray | list,
    *,
    item_names: list[str] | None = None,
    threshold: float = 1.5,
) -> DIFResult:
    """Delta plot method for DIF.

    Converts item proportions to delta scale (inversions of
    z-transformed p-values), plots focal vs reference, flags
    items far from the major axis.

    Parameters
    ----------
    p_values_ref : array-like
        Item proportion correct in reference group (k items).
    p_values_focal : array-like
        Same for focal group.
    item_names : list[str], optional
    threshold : float
        Perpendicular distance threshold for flagging (default 1.5).

    Returns
    -------
    DIFResult
        method="DeltaPlot".

    References
    ----------
    Angoff, W. H. & Ford, S. F. (1973). Item-race interaction on a
    test of scholastic aptitude. Journal of Educational Measurement.
    """
    p_r = np.clip(np.asarray(p_values_ref, dtype=np.float64), 0.001, 0.999)
    p_f = np.clip(np.asarray(p_values_focal, dtype=np.float64), 0.001, 0.999)
    k = len(p_r)

    if item_names is None:
        item_names = [f"item_{j}" for j in range(k)]

    delta_r = 4 * sp.norm.ppf(1 - p_r) + 13
    delta_f = 4 * sp.norm.ppf(1 - p_f) + 13

    slope, intercept = np.polyfit(delta_r, delta_f, 1)

    rows = []
    flagged = []
    for j in range(k):
        predicted = slope * delta_r[j] + intercept
        residual = delta_f[j] - predicted
        perp_dist = abs(residual) / np.sqrt(1 + slope**2)
        rows.append(
            {
                "item": item_names[j],
                "delta_ref": float(delta_r[j]),
                "delta_focal": float(delta_f[j]),
                "perp_distance": float(perp_dist),
            }
        )
        if perp_dist > threshold:
            flagged.append(item_names[j])

    return DIFResult(method="DeltaPlot", items=pd.DataFrame(rows), flagged=flagged)


delta_plot_dif = dif_delta_plot


def cheatsheet() -> str:
    return "dif_delta_plot({}) -> Delta plot method for DIF detection."
