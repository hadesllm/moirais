"""Baseline-conditional categorical gentrification coding.

Adapted from Laniyonu (2018) Urban Affairs Review 54(5):898–930, which
adapts Chapple / Freeman / Maciag.  The key insight: continuous
gentrification indices conflate two distinct populations -- already-
affluent tracts (immune to gentrification by construction) and
marginalised tracts that DID or DID NOT change.  The cleanest
comparator is the marginalised-but-did-not-gentrify tract.

The function returns a three-level factor:

  * ``ineligible`` -- tract was above the baseline-marginalisation
    cutoff at t=0 (top-50% on income+rent), so cannot meaningfully
    "gentrify".  Drop from analyses that want the gentrification
    comparator.
  * ``eligible``   -- tract was below the cutoff at t=0 AND did NOT
    cross the gentrification threshold by t=1.  This is the control
    group: marginalised, did-not-change.
  * ``gentrified`` -- tract was below the cutoff at t=0 AND DID cross
    the gentrification threshold (top-tercile growth in college share
    AND top-tercile growth in median rent).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


def gentrification_panel(
    df: pd.DataFrame,
    *,
    baseline_income_col: str,
    baseline_rent_col: str,
    growth_college_col: str,
    growth_rent_col: str,
    baseline_marginalisation_quantile: float = 0.5,
    gentrification_growth_quantile: float = 0.667,
    return_thresholds: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, float]]:
    """Construct a baseline-conditional 3-level gentrification factor.

    Parameters
    ----------
    df : pd.DataFrame
        One row per tract.  Must contain the four columns named below.
    baseline_income_col, baseline_rent_col : str
        Income + rent at the baseline period (e.g. 2000 census).
    growth_college_col : str
        Growth in college / BA-share between baseline and follow-up.
    growth_rent_col : str
        Growth in median rent between baseline and follow-up.
    baseline_marginalisation_quantile : float, default 0.5
        Tract is "eligible" to gentrify if baseline income AND baseline
        rent are at or below this quantile.  Default mirrors Laniyonu
        (2018) -- bottom half on both.
    gentrification_growth_quantile : float, default 0.667
        Tract "gentrified" if BOTH growth-college AND growth-rent are
        above this quantile across the panel.  Default mirrors
        Laniyonu (2018) -- top tercile on both.
    return_thresholds : bool, default False
        If True, also returns the computed cut-point values for the
        four conditions (useful for documenting the operationalisation).

    Returns
    -------
    pd.Series of object dtype with three categorical levels:
        ``"ineligible"``, ``"eligible"``, ``"gentrified"``.
        Index matches ``df.index``.

    Notes
    -----
    The dichotomous baseline cut (50%) + dichotomous growth cut (67%)
    are *both* configurable.  Laniyonu (2018) reports five
    sensitivity-of-coding specifications; rerun with different
    quantiles to mirror them.
    """
    inc_q = df[baseline_income_col].quantile(baseline_marginalisation_quantile)
    rent_q = df[baseline_rent_col].quantile(baseline_marginalisation_quantile)
    coll_q = df[growth_college_col].quantile(gentrification_growth_quantile)
    growth_rent_q = df[growth_rent_col].quantile(gentrification_growth_quantile)

    eligible_mask = (df[baseline_income_col] <= inc_q) & (df[baseline_rent_col] <= rent_q)
    growth_mask = (df[growth_college_col] >= coll_q) & (df[growth_rent_col] >= growth_rent_q)

    label = np.where(
        ~eligible_mask,
        "ineligible",
        np.where(growth_mask, "gentrified", "eligible"),
    )
    out = pd.Series(label, index=df.index, name="gentrification_panel", dtype="object")
    if return_thresholds:
        return out, {
            "baseline_income_cut": float(inc_q),
            "baseline_rent_cut": float(rent_q),
            "growth_college_cut": float(coll_q),
            "growth_rent_cut": float(growth_rent_q),
        }
    return out
