# moirais.fn — function file (hadesllm/moirais)
"""Latent mean differences between groups (requires scalar invariance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp

from moirais.fn._cfa_engine import get_mapq_structure


def mi_latent_means(data: pd.DataFrame, group_col: str, structure: dict[str, list[str]] | None = None, cdf=None, *, items: list[str] | None = None, reference_group: str | None = None) -> dict:
    """Latent mean differences between groups.

    Approximated via subscale mean differences (valid under scalar invariance).
    Reports Cohen's d for each factor and group comparison.

    Parameters
    ----------
    data : DataFrame
        Item response data with a grouping column.
    group_col : str
        Column name for the grouping variable.
    structure : dict, optional
        Factor name -> item names. Default: MAPQ 4-factor.
    items : list of str, optional
        Override item names.
    reference_group : str, optional
        Reference group (set to 0). If None, uses first sorted group.

    Returns
    -------
    dict
        Keys: reference, comparisons (list of dicts with group, factor,
        mean_diff, se, cohens_d, p_value).

    References
    ----------
    Hancock, G.R. (2001). Effect size, power, and sample size determination
        for structured means modeling. SEM, 8(2), 137-151.
    """
    if structure is None:
        structure, _ = get_mapq_structure(items)

    groups = sorted(data[group_col].dropna().unique())
    if reference_group is None:
        reference_group = str(groups[0])

    ref_data = data[data[group_col] == reference_group]
    comparisons = []

    for g in groups:
        if str(g) == str(reference_group):
            continue
        g_data = data[data[group_col] == g]

        for fname, fitems in structure.items():
            cols = [c for c in fitems if c in data.columns]
            if not cols:
                continue

            ref_scores = ref_data[cols].mean(axis=1).dropna()
            g_scores = g_data[cols].mean(axis=1).dropna()

            n_ref = len(ref_scores)
            n_g = len(g_scores)
            if n_ref < 2 or n_g < 2:
                continue

            mean_diff = float(g_scores.mean() - ref_scores.mean())
            # Pooled SD for Cohen's d
            s_ref = float(ref_scores.std(ddof=1))
            s_g = float(g_scores.std(ddof=1))
            pooled_sd = np.sqrt(((n_ref - 1) * s_ref**2 + (n_g - 1) * s_g**2) / (n_ref + n_g - 2))
            cohens_d = mean_diff / pooled_sd if pooled_sd > 1e-10 else 0.0
            se = pooled_sd * np.sqrt(1 / n_ref + 1 / n_g)
            t_stat = mean_diff / se if se > 1e-10 else 0.0
            p_val = float(2 * (1 - sp.t.cdf(abs(t_stat), n_ref + n_g - 2)))

            comparisons.append(
                {
                    "group": str(g),
                    "factor": fname,
                    "mean_diff": mean_diff,
                    "se": float(se),
                    "cohens_d": float(cohens_d),
                    "p_value": p_val,
                }
            )

    return {
        "reference": str(reference_group),
        "comparisons": comparisons,
    }


def cheatsheet() -> str:
    return "mi_latent_means({}) -> Latent mean differences between groups (requires scalar inva"
