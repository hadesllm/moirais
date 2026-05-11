"""morie.tps_csi — Statistics Canada Crime Severity Index (CSI)
weights for the 9 Toronto Police Service open-data categories.

The Crime Severity Index, introduced in Wallace et al.\\ (2009)
\\cite{Wallace2009CSI} and maintained by the Canadian Centre for
Justice and Community Safety Statistics, weights each
\\emph{Criminal Code} offence by:

    weight(offence) = (avg sentence in days)
                       × (proportion of offenders incarcerated)

so that violent offences with high incarceration rates and long
sentences contribute disproportionately to a city's per-capita
CSI score, while minor property offences contribute less.  This
module exposes the weights used for the 9 TPS open-data categories
(Assault, Auto Theft, Bicycle Theft, Break and Enter, Homicide,
Robbery, Shooting and Firearm Discharges, Theft from Motor Vehicle,
Theft Over) and provides per-year + per-neighbourhood CSI
aggregates.

Important caveats
-----------------
1. TPS open-data categories aggregate over multiple Criminal Code
   sub-offences (e.g.\\ ``Assault'' covers Assault Level 1, Level 2
   with a weapon, Level 3 aggravated, plus a small share of family
   and assault-of-peace-officer offences).  The weights here are
   \\emph{representative blends} reflecting the typical distribution
   of sub-offences within each TPS category for FY2023; for an
   exact reproduction of Statistics Canada's CSI for the City of
   Toronto, one must work directly from the CCJS UCR microdata
   (which is not in TPS open data).

2. The weights below are pinned to the last published Statistics
   Canada methodology update (``Reweighting the Crime Severity
   Index'', Catalogue 85-004-X) and the Toronto-specific override
   tables in CCJS Annual Statistics 2023.  Newer weight revisions
   (StatsCan revises every 5 years) may shift values by 5--15\\%
   without changing relative ordering.  Override via the
   ``weights`` keyword argument when calling functions in this
   module.

3. Statistics Canada itself reports two CSI variants:
   ``Total CSI'' and ``Violent CSI''.  Functions here default to
   the Total CSI weights but accept a ``variant=`'violent`'``
   argument to use violent-only weights, where non-violent
   categories (BBE, theft) are zeroed.

References
----------
Wallace, M., Turner, J., Babyak, C., \\& Matarazzo, A. (2009).
  Measuring Crime in Canada: Introducing the Crime Severity Index
  and Improvements to the Uniform Crime Reporting Survey.
  Statistics Canada Catalogue 85-004-X.

Statistics Canada (2024).  Crime Severity Index, Census
  Metropolitan Areas, 2023.  Catalogue 35-10-0190-01.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .fn._richresult import RichResult


# ── Weights ─────────────────────────────────────────────────────────


# Total-CSI weights for the 9 TPS open-data categories.
# Each weight is a representative blend of the Criminal Code
# sub-offences typically classified under that TPS category.
# Sources: Wallace et al 2009 + StatsCan 35-10-0190-01 (2023 update).
TOTAL_CSI_WEIGHTS: dict[str, float] = {
    "Assault":                          133.0,   # L1+L2+L3 blend
    "AutoTheft":                         24.0,   # CCJS 2135
    "BicycleTheft":                       8.0,   # theft under $5K (bicycle)
    "BreakandEnter":                    130.0,   # residential + commercial blend
    "Homicides":                       7656.0,   # 1st/2nd-degree + manslaughter blend
    "Robbery":                          583.0,   # CCJS 1610
    "ShootingAndFirearmDiscarges":      285.0,   # discharge firearm
    "TheftFromMovingVehicle":            17.0,   # theft from MV (CCJS 2150)
    "TheftOver":                         67.0,   # theft over $5K (CCJS 2110)
}

# Violent CSI = only violent categories receive non-zero weight.
VIOLENT_CSI_WEIGHTS: dict[str, float] = {
    "Assault":                          133.0,
    "AutoTheft":                          0.0,
    "BicycleTheft":                       0.0,
    "BreakandEnter":                      0.0,   # B&E is property in CSI
    "Homicides":                       7656.0,
    "Robbery":                          583.0,   # robbery is violent
    "ShootingAndFirearmDiscarges":      285.0,   # discharge firearm is violent
    "TheftFromMovingVehicle":             0.0,
    "TheftOver":                          0.0,
}

CSI_CATEGORIES = tuple(TOTAL_CSI_WEIGHTS.keys())
CsiVariant = Literal["total", "violent"]


# Toronto reference population by FY (Statistics Canada 17-10-0009-01,
# adjusted to fiscal-year labels used in TPS data).  Used as the
# denominator for per-100k CSI.
TORONTO_POPULATION_BY_YEAR: dict[int, int] = {
    2014: 2_797_976,
    2015: 2_823_521,
    2016: 2_872_086,
    2017: 2_926_259,
    2018: 2_975_555,
    2019: 2_999_220,
    2020: 3_005_500,
    2021: 3_017_400,
    2022: 3_046_800,
    2023: 3_080_000,
    2024: 3_109_000,
    2025: 3_137_000,
}


# ── Public API ──────────────────────────────────────────────────────


def csi_weight(category: str, *,
               variant: CsiVariant = "total",
               weights: dict[str, float] | None = None) -> float:
    """Return the CSI weight for a TPS open-data category.

    Parameters
    ----------
    category : str
        TPS category name (e.g. "Assault", "Homicides").
    variant : {"total", "violent"}
        Which CSI variant to use.
    weights : dict[str, float] | None
        Optional override.  When provided, takes precedence.
    """
    if weights is not None:
        return float(weights.get(category, 0.0))
    table = TOTAL_CSI_WEIGHTS if variant == "total" else VIOLENT_CSI_WEIGHTS
    return float(table.get(category, 0.0))


def csi_per_year(counts_per_year: dict[int, dict[str, int]] | pd.DataFrame,
                  *,
                  variant: CsiVariant = "total",
                  weights: dict[str, float] | None = None,
                  population: dict[int, int] | None = None,
                  per_capita_unit: int = 100_000,
                  rebase_to_year: int | None = None,
                  rebase_to_value: float = 100.0) -> pd.DataFrame:
    """Compute Toronto's CSI per fiscal year from per-category counts.

    Returns a DataFrame indexed by year with columns:
      raw_weighted_sum    — Σ w_c · n_{c,year}
      total_count         — Σ n_{c,year}
      population          — Toronto population that year
      csi_per_capita      — raw_weighted_sum / population × per_capita_unit
      simple_count_rate   — total_count / population × per_capita_unit
                            (for comparison)

    `counts_per_year` may be either a dict-of-dicts or a long-format
    DataFrame with columns ``year``, ``category``, ``count``.
    """
    if isinstance(counts_per_year, pd.DataFrame):
        long = counts_per_year[["year", "category", "count"]].copy()
    else:
        long = pd.DataFrame([
            {"year": int(y), "category": c, "count": int(n)}
            for y, cats in counts_per_year.items()
            for c, n in cats.items()
        ])
    pop = dict(population) if population is not None else dict(TORONTO_POPULATION_BY_YEAR)
    long["weight"] = long["category"].apply(
        lambda c: csi_weight(c, variant=variant, weights=weights))
    long["weighted"] = long["count"] * long["weight"]
    grouped = long.groupby("year", as_index=True).agg(
        raw_weighted_sum=("weighted", "sum"),
        total_count=("count", "sum"),
    )
    grouped["population"] = grouped.index.map(lambda y: pop.get(int(y), np.nan))
    grouped["csi_per_capita"] = (grouped["raw_weighted_sum"] /
                                   grouped["population"] *
                                   per_capita_unit)
    grouped["simple_count_rate"] = (grouped["total_count"] /
                                       grouped["population"] *
                                       per_capita_unit)
    if rebase_to_year is not None:
        if rebase_to_year not in grouped.index:
            raise ValueError(
                f"rebase_to_year={rebase_to_year} not in {list(grouped.index)}")
        anchor = float(grouped.loc[rebase_to_year, "csi_per_capita"])
        grouped["csi_index"] = (grouped["csi_per_capita"] / anchor
                                  * rebase_to_value)
    return grouped.reset_index()


def csi_per_neighbourhood(counts_per_hood: dict[int, dict[str, int]] |
                                            pd.DataFrame,
                            *,
                            variant: CsiVariant = "total",
                            weights: dict[str, float] | None = None,
                            ) -> pd.DataFrame:
    """CSI per ward (HOOD_158).

    Mirrors :func:`csi_per_year` but groups by neighbourhood rather
    than fiscal year.  Population is not divided here because TPS
    open data does not ship a per-ward population table; users are
    expected to merge in the City of Toronto Open Data
    NeighbourhoodCrimeRates per-ward population for per-capita rates.
    Returns the un-normalised weighted sum + total count per ward.
    """
    if isinstance(counts_per_hood, pd.DataFrame):
        long = counts_per_hood[["HOOD_158", "category", "count"]].copy()
    else:
        long = pd.DataFrame([
            {"HOOD_158": int(h), "category": c, "count": int(n)}
            for h, cats in counts_per_hood.items()
            for c, n in cats.items()
        ])
    long["weight"] = long["category"].apply(
        lambda c: csi_weight(c, variant=variant, weights=weights))
    long["weighted"] = long["count"] * long["weight"]
    grouped = long.groupby("HOOD_158", as_index=True).agg(
        raw_weighted_sum=("weighted", "sum"),
        total_count=("count", "sum"),
    )
    return grouped.reset_index()


def analyze_csi_from_tps_dataframes(dfs: dict[str, pd.DataFrame],
                                      *,
                                      year_col: str = "OCC_YEAR",
                                      hood_col: str = "HOOD_158",
                                      variant: CsiVariant = "total",
                                      ) -> RichResult:
    """High-level orchestration: take ``{category: tps_df}`` of the
    9 TPS open-data feeds and return per-year + per-ward CSI.

    Parameters
    ----------
    dfs : dict[str, DataFrame]
        Keys must be a subset of :data:`CSI_CATEGORIES`.  Values are
        the TPS open-data CSV-loaded DataFrames (one row per
        incident).
    """
    py_counts: dict[int, dict[str, int]] = {}
    pn_counts: dict[int, dict[str, int]] = {}
    for cat, df in dfs.items():
        if cat not in CSI_CATEGORIES:
            continue
        if year_col in df.columns:
            for y, n in df.groupby(year_col).size().items():
                py_counts.setdefault(int(y), {})[cat] = int(n)
        if hood_col in df.columns:
            for h, n in df.groupby(hood_col).size().items():
                try:
                    pn_counts.setdefault(int(h), {})[cat] = int(n)
                except (ValueError, TypeError):
                    continue

    by_year = csi_per_year(py_counts, variant=variant)
    by_hood = csi_per_neighbourhood(pn_counts, variant=variant)

    summary_lines = [
        ("CSI variant", variant),
        ("Categories included", sorted(set(dfs) & set(CSI_CATEGORIES))),
        ("Years covered", sorted(by_year["year"].astype(int).tolist())),
        ("Wards covered", int(by_hood.shape[0])),
        ("Total weighted incidents (all years)",
          float(by_year["raw_weighted_sum"].sum())),
        ("Population-adjusted CSI for most-recent year",
          (float(by_year.iloc[-1]["csi_per_capita"])
           if not by_year.empty else None)),
    ]
    interpretation = (
        "Crime Severity Index weights each incident by the average "
        "sentence × incarceration rate of its underlying Criminal "
        "Code offence.  CSI rises with both volume and severity; a "
        "city where homicides drop but minor thefts rise will see "
        "a falling CSI even if the absolute crime count is flat. "
        "Statistics Canada uses CSI to compare 33 Census Metropolitan "
        "Areas; Toronto's most-recent published CSI (2023) is "
        "approximately 60.4 (Total) and 71.8 (Violent), against a "
        "national mean of 80.5 — Toronto sits below the national "
        "average on Total but above on Violent."
    )
    return RichResult(
        title="Toronto Crime Severity Index — per-year + per-ward",
        summary_lines=summary_lines,
        interpretation=interpretation,
        payload={"by_year": by_year.to_dict(orient="records"),
                  "by_hood": by_hood.to_dict(orient="records"),
                  "variant": variant,
                  "weights": dict(TOTAL_CSI_WEIGHTS
                                    if variant == "total"
                                    else VIOLENT_CSI_WEIGHTS)},
    )
