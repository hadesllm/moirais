"""Biomass estimation. 'The Green speaks through me.' -- Swamp Thing"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._containers import DescriptiveResult
from ._helpers import _validate_df


def biomass_estimate(
    data: pd.DataFrame,
    *,
    dbh: str = "dbh",
    height: str = "height",
    species: str | None = None,
    a: float = 0.0673,
    b: float = 1.78,
    c: float = 0.207,
    d: float = 0.0,
) -> DescriptiveResult:
    """Allometric biomass estimation from tree diameter and height.

    Uses the general allometric equation: B = a * (DBH^b) * (H^c) * exp(d)
    Default coefficients are for tropical forests (Chave et al. 2014).

    Parameters
    ----------
    data : DataFrame
        Tree measurement data.
    dbh : str
        Diameter at breast height column (cm).
    height : str
        Tree height column (m).
    species : str or None
        Optional species grouping column.
    a, b, c, d : float
        Allometric coefficients.

    Returns
    -------
    DescriptiveResult
        ``value`` = total estimated biomass (kg).
    """
    _validate_df(data, dbh, height)
    df = data.dropna(subset=[dbh, height])
    D = df[dbh].to_numpy(dtype=float)
    H = df[height].to_numpy(dtype=float)
    if np.any(D <= 0) or np.any(H <= 0):
        raise ValueError("DBH and height must be positive")
    biomass = a * (D**b) * (H**c) * np.exp(d)
    total = float(np.sum(biomass))
    result_extra = {
        "n_trees": len(D),
        "total_biomass_kg": total,
        "mean_biomass_kg": float(np.mean(biomass)),
        "median_biomass_kg": float(np.median(biomass)),
        "sd_biomass_kg": float(np.std(biomass, ddof=1)) if len(biomass) > 1 else 0.0,
        "coefficients": {"a": a, "b": b, "c": c, "d": d},
        "total_carbon_kg": total * 0.47,
    }
    if species is not None and species in df.columns:
        sp_totals = {}
        for sp, grp in df.groupby(species):
            sp_D = grp[dbh].to_numpy(dtype=float)
            sp_H = grp[height].to_numpy(dtype=float)
            sp_bio = a * (sp_D**b) * (sp_H**c) * np.exp(d)
            sp_totals[str(sp)] = float(np.sum(sp_bio))
        result_extra["species_biomass"] = sp_totals
    return DescriptiveResult(
        name="Allometric biomass estimate",
        value=total,
        extra=result_extra,
    )


swmpt = biomass_estimate


def cheatsheet() -> str:
    return "biomass_estimate({}) -> Biomass estimation. 'The Green speaks through me.' -- Swamp "
