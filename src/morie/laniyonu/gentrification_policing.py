"""Reproducible replication of Laniyonu (2018) UAR 54(5):898–930.

"Coffee Shops and Street Stops: Policing Practices in Gentrifying
Neighborhoods" — a Spatial Durbin Model (SDM) decomposing the
direct/indirect/total effect of gentrification on NYPD stop-and-frisk
rates at the census-tract × year level.

Headline finding in the paper: gentrification has roughly zero
*direct* effect on stops/capita inside the gentrifying tract, but a
+51% to +90% *indirect* (spillover) effect on stops/capita in
neighboring tracts.  The OLS or non-spatial-FE specification would
miss this entirely.

This module is a thin user-facing wrapper.  Heavy lifting lives in
two :mod:`morie.mrm_primitives` callables:

  1. :func:`~morie.mrm_primitives.gentrification_panel` — builds the
     3-level baseline-conditional gentrification factor
     (ineligible / eligible-no-change / gentrified).
  2. :func:`~morie.mrm_primitives.spatial_spillover_decomposition` —
     given a fitted SDM's rho + beta vectors, returns the
     direct/indirect/total decomposition.

We deliberately do NOT fit the SDM ourselves: ``pysal`` / ``spreg``
is a hefty optional dep.  Callers provide either (a) the fitted
spatial-econometrics output, or (b) a configuration that triggers a
pure-numpy fallback ML estimator we ship for the SDM lite case.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..mrm_primitives.gentrification import gentrification_panel
from ..mrm_primitives.spatial_spillover import (
    SpilloverDecomposition,
    morans_i,
    spatial_spillover_decomposition,
)


@dataclass
class GentrificationPolicingResult:
    """Per-year SDM decomposition + diagnostics + reproduction notes."""
    year: int
    n_tracts: int
    rho: float
    moran_i_ols: float
    decompositions: list[SpilloverDecomposition]
    gentrification_distribution: dict[str, int]
    sensitivity_thresholds: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def interpret(self) -> str:
        gent = next(
            (d for d in self.decompositions if d.coefficient.startswith("gent")),
            None,
        )
        head = (
            f"Year {self.year}, N={self.n_tracts} tracts, rho={self.rho:+.4f}, "
            f"Moran's I (OLS residuals)={self.moran_i_ols:+.4f}.  "
        )
        gent_line = (
            f"Gentrification effect: direct={gent.direct:+.4f}, "
            f"indirect={gent.indirect:+.4f}, total={gent.total:+.4f}."
            if gent is not None
            else "Gentrification coefficient not found in decomposition."
        )
        return head + gent_line


def gentrification_policing(
    df: pd.DataFrame,
    *,
    year_col: str = "year",
    tract_id_col: str = "tract_id",
    stops_col: str = "stops",
    population_col: str = "population",
    crime_col: str = "felony_count",
    demand_col: str = "calls_311_omp",
    baseline_income_col: str = "median_inc_2000",
    baseline_rent_col: str = "median_rent_2000",
    growth_college_col: str | None = None,
    growth_rent_col: str | None = None,
    follow_income_col: str = "median_inc_2014",
    follow_rent_col: str = "median_rent_2014",
    baseline_college_col: str = "pct_ba_2000",
    follow_college_col: str = "pct_ba_2014",
    additional_controls: list[str] | None = None,
    weight_matrix: np.ndarray | None = None,
    weight_matrix_kind: Literal["queen", "knn"] = "queen",
    fitted_rho: float | dict[int, float] | None = None,
    fitted_beta_direct: dict[int, np.ndarray] | None = None,
    fitted_beta_spatial: dict[int, np.ndarray] | None = None,
    years: list[int] | None = None,
    log_outcome: bool = True,
) -> list[GentrificationPolicingResult]:
    """Replicate the Laniyonu (2018) SDM by year, return per-year decompositions.

    Two modes:

    1. **Pre-fitted mode (preferred).**  Pass ``fitted_rho``,
       ``fitted_beta_direct``, ``fitted_beta_spatial`` from your own
       SDM fit (``spreg``, ``pysal``, R's ``spatialreg``, etc.).
       This function does the diagnostic ladder + spillover
       decomposition + gentrification-coding sensitivity report.

    2. **Pure-numpy lite mode.**  When ``fitted_*`` aren't supplied,
       we run OLS, compute Moran's I on residuals, and report the
       decomposition with rho=0 as a fall-back.  Useful for sanity
       checks; for paper-ready results you almost certainly want
       mode 1.

    Parameters
    ----------
    df : pd.DataFrame
        Tract-year panel.  One row per tract per year.
    year_col, tract_id_col, stops_col, population_col,
    crime_col, demand_col : str
        Column names.  Defaults match the schema documented in
        ROADMAP.md's ``load_gentrification_nyc()`` toy bundle.
    baseline_income_col, baseline_rent_col : str
        Income + rent at baseline period (2000 in the paper).
    growth_college_col, growth_rent_col : str or None
        Growth columns.  If None, computed from
        ``follow_<x>_col`` minus ``baseline_<x>_col``.
    follow_income_col, follow_rent_col, baseline_college_col,
    follow_college_col : str
        Used when growth columns are not pre-computed.
    additional_controls : list[str], optional
        Extra tract-year controls (pct_black, pct_latino, etc.).
    weight_matrix : np.ndarray, optional
        Pre-computed (N, N) row-standardised spatial weights.
        Required when ``fitted_*`` mode is in use.  In lite mode,
        if None, we synthesise a uniform-block W (placeholder).
    weight_matrix_kind : "queen" | "knn"
        Provenance label only; we don't recompute W here.
    fitted_rho, fitted_beta_direct, fitted_beta_spatial :
        Pass these to bypass the lite-mode fall-back.
    years : list[int], optional
        Subset of years to analyse.  Default = all years in df.
    log_outcome : bool, default True
        If True, the outcome is ``log(stops / population)``.  If
        False, raw ``stops / population``.

    Returns
    -------
    list[GentrificationPolicingResult], one per year analysed.
    """
    # 1. Build the gentrification panel from the baseline-period rows.
    #    The categorical flag is constant across years within a tract.
    if growth_college_col is None:
        df = df.copy()
        df["_growth_college"] = (
            df[follow_college_col] - df[baseline_college_col]
        )
        growth_college_col = "_growth_college"
    if growth_rent_col is None:
        df["_growth_rent"] = df[follow_rent_col] - df[baseline_rent_col]
        growth_rent_col = "_growth_rent"

    # Build gentrification flag on the tract-level baseline frame
    baseline_frame = df.drop_duplicates(tract_id_col).set_index(tract_id_col)
    gent_flag, thresholds = gentrification_panel(
        baseline_frame,
        baseline_income_col=baseline_income_col,
        baseline_rent_col=baseline_rent_col,
        growth_college_col=growth_college_col,
        growth_rent_col=growth_rent_col,
        return_thresholds=True,
    )
    df = df.set_index(tract_id_col).join(
        gent_flag.rename("gentrification"), how="left"
    ).reset_index()

    gent_dist_overall = (
        df.drop_duplicates(tract_id_col)["gentrification"]
        .value_counts()
        .to_dict()
    )

    # 2. Per-year SDM decomposition
    if years is None:
        years = sorted(df[year_col].dropna().unique().tolist())

    out: list[GentrificationPolicingResult] = []
    for yr in years:
        yr_df = df[df[year_col] == yr].copy()
        n = len(yr_df)
        if n < 10:
            warnings.warn(
                f"morie.laniyonu.gentrification_policing: year {yr} has "
                f"only {n} tracts; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Build design matrix X and outcome y
        gent_dummies = pd.get_dummies(
            yr_df["gentrification"], prefix="gent", drop_first=True
        ).astype(float)
        X_cols = list(gent_dummies.columns) + [crime_col, demand_col]
        if additional_controls:
            X_cols += additional_controls
        X = pd.concat(
            [gent_dummies, yr_df[[crime_col, demand_col] + (additional_controls or [])]],
            axis=1,
        ).to_numpy(dtype=float)
        rate = yr_df[stops_col] / yr_df[population_col].clip(lower=1)
        y = np.log(rate.clip(lower=1e-10)).to_numpy() if log_outcome else rate.to_numpy()

        # OLS diagnostic — used to seed Moran's I and lite-mode beta
        X_int = np.column_stack([np.ones(n), X])
        try:
            beta_ols = np.linalg.lstsq(X_int, y, rcond=None)[0]
            residuals = y - X_int @ beta_ols
        except np.linalg.LinAlgError:
            beta_ols = np.zeros(X_int.shape[1])
            residuals = y - y.mean()

        # Moran's I requires a W of the right size
        if weight_matrix is None:
            # Lite-mode placeholder: a row-standardised k=4 NN-like dense W
            # built from felony-count similarity.  NOT a real contiguity W;
            # in pre-fitted mode the caller supplies the real one.
            W = _placeholder_weight_matrix(yr_df[crime_col].to_numpy(dtype=float))
            note = ("Using placeholder W from felony-count proximity; "
                    "pass weight_matrix=... for paper-grade results.")
        else:
            W = weight_matrix
            note = f"Using user-supplied W (kind={weight_matrix_kind})."

        try:
            mi = morans_i(residuals, W)
        except ValueError:
            mi = float("nan")

        # SDM decomposition — pre-fitted or lite-mode fall-back
        if (fitted_rho is not None and fitted_beta_direct is not None
                and fitted_beta_spatial is not None):
            rho_yr = (fitted_rho[yr] if isinstance(fitted_rho, dict)
                      else float(fitted_rho))
            beta_d = fitted_beta_direct[yr]
            beta_s = fitted_beta_spatial[yr]
        else:
            rho_yr = 0.0
            beta_d = beta_ols[1:]                 # drop intercept
            beta_s = np.zeros_like(beta_d)
            note += "  (lite mode: rho=0; pass fitted_* for SDM decomposition.)"

        decomps = spatial_spillover_decomposition(
            rho=rho_yr,
            beta_direct=beta_d,
            beta_spatial=beta_s,
            W=W,
            coefficient_names=X_cols,
        )

        gent_dist_yr = (
            yr_df["gentrification"].value_counts().to_dict()
        )

        out.append(GentrificationPolicingResult(
            year=int(yr),
            n_tracts=n,
            rho=float(rho_yr),
            moran_i_ols=float(mi),
            decompositions=decomps,
            gentrification_distribution=gent_dist_yr,
            sensitivity_thresholds=thresholds,
            note=note,
        ))

    return out


def _placeholder_weight_matrix(crime_arr: np.ndarray, k: int = 4) -> np.ndarray:
    """k-NN-style row-standardised W from crime-count proximity.

    Stand-in for a real geographic contiguity W when the caller hasn't
    supplied one.  Not paper-grade — Moran's I computed against this
    will be sensible-but-not-canonical.
    """
    n = crime_arr.size
    if n < 2:
        return np.zeros((n, n))
    k = min(k, n - 1)
    W = np.zeros((n, n))
    for i in range(n):
        diffs = np.abs(crime_arr - crime_arr[i])
        diffs[i] = np.inf
        neighbors = np.argsort(diffs)[:k]
        W[i, neighbors] = 1.0
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums
