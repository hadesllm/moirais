# morie.fn -- function file (hadesllm/morie)
"""Wright map data (item difficulty vs person ability)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def irt_wright_map(
    item_params: dict,
    theta: np.ndarray,
) -> dict:
    """Prepare Wright map data: item difficulties and person abilities.

    Parameters
    ----------
    item_params : dict
        {item_name: {'b': ...}} or {item_name: {'a': ..., 'b': ...}}.
    theta : ndarray
        Person ability estimates (n,).

    Returns
    -------
    dict
        Keys: 'items' (DataFrame with item, b), 'persons' (dict with
        theta array and descriptive stats), 'alignment' (overlap stats).

    References
    ----------
    Wright, B. D. & Stone, M. H. (1979). Best Test Design. MESA Press.
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()

    # Item difficulties
    item_rows = []
    for item, params in item_params.items():
        b = params.get("b", np.nan)
        if isinstance(b, (list, np.ndarray)):
            for idx, bval in enumerate(b):
                item_rows.append({"item": f"{item}_t{idx + 1}", "b": float(bval)})
        else:
            item_rows.append({"item": str(item), "b": float(b)})
    items_df = pd.DataFrame(item_rows)

    # Person stats
    valid_theta = theta[~np.isnan(theta)]
    person_stats = {
        "n": len(valid_theta),
        "mean": float(np.mean(valid_theta)) if len(valid_theta) > 0 else np.nan,
        "sd": float(np.std(valid_theta, ddof=1)) if len(valid_theta) > 1 else np.nan,
        "min": float(np.min(valid_theta)) if len(valid_theta) > 0 else np.nan,
        "max": float(np.max(valid_theta)) if len(valid_theta) > 0 else np.nan,
    }

    # Alignment: proportion of persons within item difficulty range
    b_vals = items_df["b"].dropna().values
    if len(b_vals) > 0 and len(valid_theta) > 0:
        b_min, b_max = float(b_vals.min()), float(b_vals.max())
        in_range = float(np.mean((valid_theta >= b_min) & (valid_theta <= b_max)))
    else:
        b_min = b_max = np.nan
        in_range = np.nan

    return {
        "items": items_df,
        "persons": {"theta": theta, **person_stats},
        "alignment": {
            "item_range": [b_min, b_max],
            "pct_persons_in_range": in_range,
        },
    }


def cheatsheet() -> str:
    return "irt_wright_map({}) -> Wright map data (item difficulty vs person ability)."
