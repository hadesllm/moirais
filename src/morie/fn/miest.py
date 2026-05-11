# morie.fn — function file (hadesllm/morie)
"""Effect sizes for measurement invariance (dMACS, signed area)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn._mapq_const import SUBSCALES


def mi_effect_size(
    fit_list: list[dict],
    *,
    data: pd.DataFrame | None = None,
    group_col: str | None = None,
    items: list[str] | None = None,
) -> pd.DataFrame:
    """Effect sizes for measurement invariance.

    Computes dMACS (Nye & Drasgow, 2011) approximation when data is provided,
    otherwise reports delta-CFI and delta-RMSEA as effect sizes.

    Parameters
    ----------
    fit_list : list of dict
        List of invariance fit dicts (from mi_by_gender or similar).
    data : DataFrame, optional
        Item response data (needed for dMACS computation).
    group_col : str, optional
        Grouping column (needed for dMACS).
    items : list of str, optional
        Item column names for dMACS.

    Returns
    -------
    DataFrame
        Columns: level, delta_cfi, delta_rmsea, effect_size_label.
        If data provided, also includes per-item dMACS.

    References
    ----------
    Nye, C.D. & Drasgow, F. (2011). Effect size indices for analyses of
        measurement equivalence. J. Applied Psychology, 96(5), 966-980.
    """
    rows = []
    for i, fit in enumerate(fit_list):
        level = fit.get("level", f"level_{i}")
        delta = fit.get("delta_fit", {})
        d_cfi = abs(delta.get("delta_cfi", 0.0))
        d_rmsea = abs(delta.get("delta_rmsea", 0.0))

        # Classify effect size
        if d_cfi <= 0.005:
            label = "negligible"
        elif d_cfi <= 0.01:
            label = "small"
        elif d_cfi <= 0.02:
            label = "moderate"
        else:
            label = "large"

        rows.append(
            {
                "level": level,
                "delta_cfi": d_cfi,
                "delta_rmsea": d_rmsea,
                "effect_size_label": label,
            }
        )

    df = pd.DataFrame(rows)

    # Compute dMACS if data provided
    if data is not None and group_col is not None:
        all_items = items if items is not None else [c for sub in SUBSCALES.values() for c in sub]
        all_items = [c for c in all_items if c in data.columns]
        groups = sorted(data[group_col].dropna().unique())

        if len(groups) >= 2:
            g1_data = data[data[group_col] == groups[0]]
            g2_data = data[data[group_col] == groups[1]]
            dmacs_rows = []
            for item in all_items:
                v1 = g1_data[item].dropna().to_numpy(dtype=np.float64)
                v2 = g2_data[item].dropna().to_numpy(dtype=np.float64)
                if len(v1) < 2 or len(v2) < 2:
                    continue
                # dMACS approximation: |mean_diff| / pooled_sd
                pooled_sd = np.sqrt(
                    ((len(v1) - 1) * np.var(v1, ddof=1) + (len(v2) - 1) * np.var(v2, ddof=1)) / (len(v1) + len(v2) - 2)
                )
                dmacs = abs(np.mean(v1) - np.mean(v2)) / pooled_sd if pooled_sd > 1e-10 else 0.0
                dmacs_rows.append({"item": item, "dmacs": float(dmacs)})

            if dmacs_rows:
                df.attrs["dmacs"] = pd.DataFrame(dmacs_rows)

    return df


def cheatsheet() -> str:
    return "mi_effect_size({}) -> Effect sizes for measurement invariance (dMACS, signed area)"
