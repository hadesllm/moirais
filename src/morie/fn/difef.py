# morie.fn — function file (hadesllm/morie)
"""DIF effect size (MH delta) with ETS classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def difef(
    mh_or: np.ndarray | pd.Series | list,
    *,
    item_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute MH delta effect sizes and ETS DIF classification.

    MH delta = -2.35 * ln(MH odds ratio), following the ETS delta
    metric convention (Holland & Thayer, 1988).

    ETS classification thresholds:
      A: |delta| < 1.0 (negligible DIF)
      B: 1.0 <= |delta| < 1.5 (moderate DIF)
      C: |delta| >= 1.5 (large DIF)

    Parameters
    ----------
    mh_or : array-like
        Mantel-Haenszel odds ratios per item (from difmh output or
        manually computed).
    item_names : list[str], optional
        Item labels.  If None, uses item_0, item_1, ...

    Returns
    -------
    DataFrame
        Columns: item, mh_or, delta, abs_delta, classification.

    References
    ----------
    Holland, P. W. & Thayer, D. T. (1988). Differential item performance
    and the Mantel-Haenszel procedure. In H. Wainer & H. I. Braun (Eds.),
    Test Validity. Lawrence Erlbaum.

    Dorans, N. J. & Holland, P. W. (1993). DIF detection and description:
    Mantel-Haenszel and standardization. In P. W. Holland & H. Wainer (Eds.),
    Differential Item Functioning. Lawrence Erlbaum.
    """
    or_arr = np.asarray(mh_or, dtype=np.float64).ravel()
    n_items = len(or_arr)

    if item_names is None:
        item_names = [f"item_{j}" for j in range(n_items)]

    if len(item_names) != n_items:
        raise ValueError(f"item_names length ({len(item_names)}) != n_items ({n_items}).")

    delta = np.full(n_items, np.nan)
    for j in range(n_items):
        if or_arr[j] > 0 and np.isfinite(or_arr[j]):
            delta[j] = -2.35 * np.log(or_arr[j])

    abs_delta = np.abs(delta)

    classification = []
    for j in range(n_items):
        ad = abs_delta[j]
        if np.isnan(ad):
            classification.append("NA")
        elif ad < 1.0:
            classification.append("A")
        elif ad < 1.5:
            classification.append("B")
        else:
            classification.append("C")

    return pd.DataFrame(
        {
            "item": item_names,
            "mh_or": or_arr,
            "delta": delta,
            "abs_delta": abs_delta,
            "classification": classification,
        }
    )


dif_effect = difef


def cheatsheet() -> str:
    return "difef({}) -> DIF effect size (MH delta) with ETS classification."
