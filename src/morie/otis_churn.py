"""morie.otis_churn — Goffmanian institutional-churn analyses on OTIS.

Goffman (1961) describes 'total institutions' as totalizing environments
that produce a *cyclical* relationship to the inmate. This module
operationalises the cyclical / mortifying / embedding dimensions as
formal statistical tests:

    repeat_placement_concentration(b09)   — Gini + power-law fit + KS test
    within_year_placement_count(b01)      — per-(id,year) cell-size distribution
    within_year_region_diversity(b01)     — intra-year region cycling
    mortification_cooccurrence(b01)       — joint alert chi² + Cramer's V
    disciplinary_medical_overlap(b01)     — disc × med-protect co-occurrence
    embedding_distribution(b02)           — total-days survival (lognormal vs Pareto AIC)
    intra_year_transition_matrix(a01)     — Markov regA→regA' within FY
    path_complexity_gini(b01)             — Gini split by (year × region)
    region_alert_state_richness(b01)      — distinct (region × combo) per cell
    regC_demog_contingency(b01)           — multi-region × Gender × Age χ²
    irr_glmm_vm(b01)                      — Poisson + NB IRR for vm

Each emits a RichResult.

NOTE: OTIS `UniqueIndividual_ID` is anonymised as `YYYY-XXXXX-AA`, where
`AA` is the dataset acronym (`RC` for a01, `SG` for b01/b02, `DC` for d01).
The YYYY prefix locks each ID to one fiscal year, AND IDs are re-randomized
per dataset file even within the same year. So longitudinal individual-level
linkage and cross-dataset linkage are both **not possible by design**.
All metrics above are either aggregate (b09 counts), within-year individual
(b01), or within-year total-days (b02). See `docs/source/methods/otis_linkage.md`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .fn._richresult import RichResult

PROJECT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = PROJECT / "data/manifest/outputs/otis_churn"


def _load(ds_id: str) -> pd.DataFrame:
    from .otis_datasets import load_otis_dataset
    return load_otis_dataset(ds_id)


# ── 1. Repeat-placement concentration ──────────────────────────────


def gini(x: np.ndarray) -> float:
    if x.size == 0 or x.sum() == 0:
        return 0.0
    sx = np.sort(x.astype(float))
    n = sx.size
    cum = np.cumsum(sx)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def _powerlaw_mle(x: np.ndarray, x_min: float = 1) -> tuple[float, float]:
    """Hill-MLE for discrete power-law: alpha = 1 + n / sum(ln(x_i / (x_min - 0.5)))
    Returns (alpha, std_err).
    """
    x = x[x >= x_min].astype(float)
    n = x.size
    if n < 5:
        return float("nan"), float("nan")
    s = np.sum(np.log(x / (x_min - 0.5)))
    alpha = 1 + n / s if s > 0 else float("nan")
    se = (alpha - 1) / np.sqrt(n) if np.isfinite(alpha) else float("nan")
    return alpha, se


def _parse_placement_bin(label: str) -> float:
    """Parse OTIS b09 placement-count bin labels.
    '1 placement' -> 1
    '2 placements' -> 2
    '6 to 10 placements' -> 8 (midpoint)
    '11 to 15 placements' -> 13 (midpoint)
    """
    import re as _re
    s = str(label).strip().lower()
    rng = _re.search(r"(\d+)\s+to\s+(\d+)", s)
    if rng:
        a, b = int(rng.group(1)), int(rng.group(2))
        return (a + b) / 2.0
    n = _re.search(r"(\d+)", s)
    return float(n.group(1)) if n else float("nan")


def repeat_placement_concentration(df: pd.DataFrame | None = None) -> RichResult:
    """Goffman's 'cyclical inmate' — heavy concentration of placements
    among a small fraction of individuals.

    Uses OTIS b09 (NumberPlacements × NumberIndividuals).
    """
    df = df if df is not None else _load("b09")
    if "NumberPlacements_Segregation" not in df.columns:
        return RichResult(title="Repeat-placement concentration",
                          warnings=["b09 columns missing"])
    # Reconstruct individual-level placement count vector
    # row = (count_value, n_indivs at that count)
    rows = df.dropna(subset=["NumberPlacements_Segregation",
                              "NumberIndividuals_Segregation"])
    counts = rows["NumberPlacements_Segregation"].apply(_parse_placement_bin).values
    multipliers = rows["NumberIndividuals_Segregation"].astype(int).values
    if counts.size == 0:
        return RichResult(title="Repeat-placement concentration",
                          warnings=["b09 has no usable rows"])
    # Expand: for each (count, n_indiv) row, repeat `count` n_indiv times
    expanded = np.repeat(counts, multipliers)
    g = gini(expanded)
    alpha, alpha_se = _powerlaw_mle(expanded, x_min=1)

    # Top-decile share
    sorted_desc = np.sort(expanded)[::-1]
    n = sorted_desc.size
    top10pct = sorted_desc[: max(1, n // 10)].sum()
    total = sorted_desc.sum()
    top10_share = top10pct / total if total > 0 else 0.0

    # KS test vs exponential
    if expanded.size >= 5 and expanded.std() > 0:
        scale = expanded.mean()
        ks = sps.kstest(expanded, "expon", args=(0, scale))
        ks_p = float(ks.pvalue)
    else:
        ks_p = float("nan")

    return RichResult(
        title="Goffmanian: repeat-placement concentration",
        summary_lines=[
            ("OTIS source", "b09 — placements per individual"),
            ("Individuals (in seg)", int(n)),
            ("Mean placements/person", round(float(expanded.mean()), 2)),
            ("Median placements/person", float(np.median(expanded))),
            ("Max placements/person", int(expanded.max())),
            ("Gini coefficient", round(g, 4)),
            ("Top-10% share of placements", f"{100*top10_share:.1f}%"),
            ("Power-law α (Hill MLE)",
                round(alpha, 3) if np.isfinite(alpha) else "n/a"),
            ("Power-law α SE",
                round(alpha_se, 3) if np.isfinite(alpha_se) else "n/a"),
            ("KS-test p-value vs exponential",
                round(ks_p, 6) if np.isfinite(ks_p) else "n/a"),
        ],
        interpretation=(
            f"Gini = {g:.3f} on placements/person. Goffmanian total-"
            "institution dynamics predict heavy-tail concentration "
            "(few individuals account for many placements). "
            f"Power-law α ≈ {alpha:.2f} — typical Goffmanian range is "
            "1.5–2.5 (preferential-attachment-style cycling). KS-test "
            f"p={ks_p:.4f} against exponential null: rejection ⇒ "
            "non-exponential heavy tail."
        ),
        payload={"gini": g, "alpha_mle": alpha, "alpha_se": alpha_se,
                 "ks_pvalue": ks_p, "top10_share": top10_share, "n": int(n)},
    )


# ── 2. Within-year placement-count distribution ───────────────────


def within_year_placement_count(df: pd.DataFrame | None = None) -> RichResult:
    """Distribution of segregation placements per (individual × fiscal
    year) cell.

    Because OTIS `UniqueIndividual_ID` is anonymised per fiscal year
    (`YYYY-XXXXX-AA`), each (id, year) cell is one anonymous person-year.
    This metric describes how concentrated the placements are *within*
    a fiscal year — Goffman's intra-year cycling. Cross-year readmission
    is **not measurable** in OTIS by design. See OTIS_LINKAGE.md.
    """
    df = df if df is not None else _load("b01")
    needed = {"UniqueIndividual_ID", "EndFiscalYear"}
    if not needed.issubset(df.columns):
        return RichResult(title="Within-year placement count",
                          warnings=["b01 missing required cols"])
    g = df.dropna(subset=["UniqueIndividual_ID", "EndFiscalYear"])
    if g.empty:
        return RichResult(title="Within-year placement count",
                          warnings=["no usable rows"])
    counts = g.groupby(["UniqueIndividual_ID", "EndFiscalYear"]).size()
    cnt_arr = counts.to_numpy()
    return RichResult(
        title="Within-year placement count (Goffmanian intra-year cycling)",
        summary_lines=[
            ("Distinct (id × year) cells", int(counts.size)),
            ("Mean placements / person-year", round(float(cnt_arr.mean()), 3)),
            ("Median", float(np.median(cnt_arr))),
            ("Q1 / Q3",
                f"{float(np.quantile(cnt_arr, 0.25)):.1f} / "
                f"{float(np.quantile(cnt_arr, 0.75)):.1f}"),
            ("Max placements in one FY", int(cnt_arr.max())),
            ("Cells with 1 placement",
                int((cnt_arr == 1).sum())),
            ("Cells with 2 placements",
                int((cnt_arr == 2).sum())),
            ("Cells with 3+ placements",
                int((cnt_arr >= 3).sum())),
            ("% with multiple placements within FY",
                f"{100*float((cnt_arr > 1).mean()):.1f}%"),
            ("Gini of placement counts", round(gini(cnt_arr), 3)),
        ],
        interpretation=(
            f"Of {counts.size} person-year cells, "
            f"{(cnt_arr > 1).sum()} ({100*float((cnt_arr > 1).mean()):.1f}%) "
            "received more than one segregation placement within the same "
            "fiscal year. Gini = "
            f"{gini(cnt_arr):.3f}. This is the intra-year concentration "
            "of segregation exposure; cross-year readmission cannot be "
            "measured because OTIS IDs are year-locked."
        ),
        payload={"n_cells": int(counts.size),
                 "mean_count": float(cnt_arr.mean()),
                 "median_count": float(np.median(cnt_arr)),
                 "max_count": int(cnt_arr.max()),
                 "frac_multi": float((cnt_arr > 1).mean()),
                 "gini": float(gini(cnt_arr))},
        warnings=[
            "OTIS UniqueIndividual_ID is year-locked (YYYY-XXXXX-AA); "
            "this metric measures intra-year placement count only.",
        ],
    )


# ── 3. Within-year region diversity ────────────────────────────────


def within_year_region_diversity(df: pd.DataFrame | None = None) -> RichResult:
    """Distribution of distinct regions per individual *within* one
    fiscal year.

    Because OTIS `UniqueIndividual_ID` is anonymised per fiscal year
    (`YYYY-XXXXX-AA`), this metric measures **intra-year** region
    cycling only — not multi-year mobility. See OTIS_LINKAGE.md.
    """
    df = df if df is not None else _load("b01")
    if "Region_AtTimeOfPlacement" not in df.columns or \
            "UniqueIndividual_ID" not in df.columns:
        return RichResult(title="Within-year region diversity",
                          warnings=["b01 missing region/individual cols"])
    g = df.dropna(subset=["UniqueIndividual_ID", "Region_AtTimeOfPlacement"])
    n_regions_per = g.groupby("UniqueIndividual_ID")["Region_AtTimeOfPlacement"].nunique()
    if n_regions_per.size == 0:
        return RichResult(title="Within-year region diversity",
                          warnings=["no usable rows"])
    return RichResult(
        title="Within-year region diversity (Goffmanian intra-year mobility)",
        summary_lines=[
            ("Distinct (id × year) cells", int(n_regions_per.size)),
            ("Mean #regions per person-year",
                round(float(n_regions_per.mean()), 3)),
            ("In just 1 region", int((n_regions_per == 1).sum())),
            ("In 2 regions",
                int((n_regions_per == 2).sum())),
            ("In 3+ regions",
                int((n_regions_per >= 3).sum())),
            ("% multi-region within FY",
                f"{100*float((n_regions_per > 1).mean()):.1f}%"),
        ],
        tables=[{
            "title": "Counts by # regions visited within one FY:",
            "headers": ["#regions", "n_person_years", "%"],
            "rows": [[int(k), int(v), f"{100*v/n_regions_per.size:.1f}%"]
                     for k, v in n_regions_per.value_counts().sort_index().items()],
        }],
        interpretation=(
            f"{100*(n_regions_per > 1).mean():.1f}% of person-years involve "
            "movement across more than one region within the same fiscal "
            "year — intra-year cross-staff-regime mobility. Multi-year "
            "mobility cannot be measured because OTIS IDs are year-locked."
        ),
        warnings=[
            "OTIS UniqueIndividual_ID is year-locked (YYYY-XXXXX-AA); "
            "this metric is intra-year only.",
        ],
    )


# ── 4. Mortification co-occurrence ────────────────────────────────


def mortification_cooccurrence(df: pd.DataFrame | None = None) -> RichResult:
    """Goffman's 'mortification of self' is operationalised here as
    concurrent stigmatising-alert flags on a single placement.
    """
    df = df if df is not None else _load("b01")
    cols = ["MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert"]
    have = [c for c in cols if c in df.columns]
    if len(have) < 2:
        return RichResult(title="Mortification co-occurrence",
                          warnings=["need ≥2 alert columns"])
    # Convert Yes/No strings to bool
    def _yn(s):
        return s.astype(str).str.strip().str.lower().eq("yes")
    flags = pd.DataFrame({c: _yn(df[c]) for c in have})
    n_flags = flags.sum(axis=1)
    counts = n_flags.value_counts().sort_index()
    n = int(flags.shape[0])

    # Chi² for independence of MentalHealth & SuicideRisk
    if "MentalHealth_Alert" in have and "SuicideRisk_Alert" in have:
        ct = pd.crosstab(flags["MentalHealth_Alert"],
                          flags["SuicideRisk_Alert"])
        try:
            chi2, p, dof, _ = sps.chi2_contingency(ct)
            cramer_v = float(np.sqrt(chi2 / (ct.values.sum() *
                                              max(1, min(ct.shape) - 1))))
        except Exception:
            chi2, p, dof, cramer_v = float("nan"), float("nan"), 0, float("nan")
    else:
        chi2, p, dof, cramer_v = float("nan"), float("nan"), 0, float("nan")

    return RichResult(
        title="Goffmanian: mortification co-occurrence",
        summary_lines=[
            ("Alerts considered", ", ".join(have)),
            ("Placements", int(n)),
            ("0 alerts", int((n_flags == 0).sum())),
            ("1 alert", int((n_flags == 1).sum())),
            ("2 alerts", int((n_flags == 2).sum())),
            ("3 alerts", int((n_flags == 3).sum())),
            ("MH × Suicide-Risk χ²", round(float(chi2), 3)),
            ("χ² p-value", round(float(p), 6)),
            ("Cramer's V", round(float(cramer_v), 4)),
        ],
        tables=[{
            "title": "# concurrent alerts per placement:",
            "headers": ["#alerts", "n_placements", "%"],
            "rows": [[int(k), int(v), f"{100*v/n:.1f}%"]
                     for k, v in counts.items()],
        }],
        interpretation=(
            f"Cramer's V = {cramer_v:.3f} between MH and Suicide-Risk "
            "alerts. Higher V = stronger co-occurrence (Goffman's "
            f"mortification stack). χ² p={p:.4g}: rejection of "
            "independence ⇒ alerts cluster on the same placements rather "
            "than being independently assigned."
        ),
    )


# ── 5. Disciplinary × medical-protection overlap ──────────────────


def disciplinary_medical_overlap(df: pd.DataFrame | None = None) -> RichResult:
    """Goffman's 'tinkering trades' tension: the same institution
    classifies the same person via *medical* AND *disciplinary*
    rationales. We look at co-occurrence on individual records.
    """
    df = df if df is not None else _load("b01")
    disc_cols = [c for c in df.columns
                 if c.startswith("SegReason_Disciplinary")]
    med_cols = [c for c in df.columns
                if c.startswith("SegReason_") and "Medical" in c]
    if not disc_cols or not med_cols:
        return RichResult(title="Disciplinary × medical overlap",
                          warnings=["missing disciplinary or medical "
                                    "SegReason columns"])
    def _yn_any(cols):
        return df[cols].apply(
            lambda s: s.astype(str).str.strip().str.lower().eq("yes")
        ).any(axis=1)
    has_disc = _yn_any(disc_cols)
    has_med = _yn_any(med_cols)
    ct = pd.crosstab(has_disc, has_med, rownames=["disc"], colnames=["med"])
    try:
        chi2, p, _, _ = sps.chi2_contingency(ct)
        n = ct.values.sum()
        cramer_v = float(np.sqrt(chi2 / (n * max(1, min(ct.shape) - 1))))
    except Exception:
        chi2, p, cramer_v = float("nan"), float("nan"), float("nan")
    return RichResult(
        title="Goffmanian: disciplinary × medical-protection overlap",
        summary_lines=[
            ("Placements", int(df.shape[0])),
            ("Disciplinary cols",
                ", ".join(c.replace('SegReason_', '') for c in disc_cols)),
            ("Medical-protection cols",
                ", ".join(c.replace('SegReason_', '') for c in med_cols)),
            ("Both flagged",
                int((has_disc & has_med).sum())),
            ("Disciplinary only",
                int((has_disc & ~has_med).sum())),
            ("Medical only",
                int((~has_disc & has_med).sum())),
            ("Neither",
                int((~has_disc & ~has_med).sum())),
            ("χ² statistic", round(float(chi2), 3)),
            ("χ² p-value", round(float(p), 6)),
            ("Cramer's V", round(float(cramer_v), 4)),
        ],
        interpretation=(
            "Goffman's 'tinkering trades' surface where punitive and "
            "therapeutic logics co-classify the same person. "
            f"Cramer's V={cramer_v:.3f} measures their dependence. "
            f"χ² p={p:.4g} — rejection ⇒ joint flagging is non-random."
        ),
    )


# ── 6. Total-days embedding distribution ──────────────────────────


def embedding_distribution(df: pd.DataFrame | None = None) -> RichResult:
    """TotalAggregatedDays_Segregation distribution per individual per
    fiscal year. Compare lognormal vs Pareto fit via AIC.
    """
    df = df if df is not None else _load("b02")
    if "TotalAggregatedDays_Segregation" not in df.columns:
        return RichResult(title="Embedding distribution",
                          warnings=["b02 missing TotalAggregatedDays"])
    days = pd.to_numeric(df["TotalAggregatedDays_Segregation"],
                          errors="coerce").dropna()
    days = days[days > 0]
    if days.size < 50:
        return RichResult(title="Embedding distribution",
                          warnings=[f"only {days.size} valid rows"])
    x = days.values.astype(float)
    # MLE fits
    try:
        ln_shape, ln_loc, ln_scale = sps.lognorm.fit(x, floc=0)
        ln_aic = 2 * 2 - 2 * sps.lognorm.logpdf(x, ln_shape,
                                                  ln_loc, ln_scale).sum()
    except Exception:
        ln_aic = float("nan")
    try:
        pa_b, pa_loc, pa_scale = sps.pareto.fit(x, floc=0)
        pa_aic = 2 * 2 - 2 * sps.pareto.logpdf(x, pa_b, pa_loc, pa_scale).sum()
    except Exception:
        pa_aic = float("nan")
    try:
        ex_loc, ex_scale = sps.expon.fit(x, floc=0)
        ex_aic = 2 * 1 - 2 * sps.expon.logpdf(x, ex_loc, ex_scale).sum()
    except Exception:
        ex_aic = float("nan")
    fits = [("lognormal", ln_aic), ("pareto", pa_aic), ("exponential", ex_aic)]
    fits_valid = [f for f in fits if np.isfinite(f[1])]
    best = min(fits_valid, key=lambda f: f[1])[0] if fits_valid else "n/a"
    return RichResult(
        title="Goffmanian: institutional embedding (total days) distribution",
        summary_lines=[
            ("Records", int(days.size)),
            ("Mean total days", round(float(days.mean()), 1)),
            ("Median", float(days.median())),
            ("Max", int(days.max())),
            ("Lognormal AIC",
                round(float(ln_aic), 1) if np.isfinite(ln_aic) else "n/a"),
            ("Pareto AIC",
                round(float(pa_aic), 1) if np.isfinite(pa_aic) else "n/a"),
            ("Exponential AIC",
                round(float(ex_aic), 1) if np.isfinite(ex_aic) else "n/a"),
            ("Best fit (lowest AIC)", best),
        ],
        interpretation=(
            f"Best fit: {best}. Pareto / lognormal beats exponential "
            "= heavy-tailed, consistent with Goffman's dichotomy between "
            "the casual short-stayer and the deeply-embedded long-stayer."
        ),
        payload={"lognormal_aic": float(ln_aic) if np.isfinite(ln_aic) else None,
                 "pareto_aic": float(pa_aic) if np.isfinite(pa_aic) else None,
                 "exponential_aic": float(ex_aic) if np.isfinite(ex_aic) else None,
                 "best_fit": best, "n": int(days.size)},
    )


# ── Master driver ──────────────────────────────────────────────────


# ── 7. Within-year region transition matrix (Markov) ──────────────


def _binarise(s: pd.Series) -> pd.Series:
    """Tolerant 'Yes/No/True/False' → 0/1."""
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0).astype(int) > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(
        {"yes", "y", "true", "t", "1"}).astype(int)


def intra_year_transition_matrix(df: pd.DataFrame | None = None) -> RichResult:
    """Knowledge itself is power. — Francis Bacon"""
    df = df if df is not None else _load("a01")
    needed = {"UniqueIndividual_ID", "EndFiscalYear",
              "Region_AtTimeOfPlacement"}
    if not needed.issubset(df.columns):
        return RichResult(title="Intra-year region transition matrix",
                          warnings=["a01 missing required cols"])
    base = df.dropna(subset=list(needed)).copy()
    base["regA"] = base["Region_AtTimeOfPlacement"].astype(str)
    base = base.sort_values(["UniqueIndividual_ID", "EndFiscalYear"])
    base["regA_prev"] = base.groupby(
        ["UniqueIndividual_ID", "EndFiscalYear"])["regA"].shift(1)
    edges = base.dropna(subset=["regA_prev"])
    if edges.empty:
        return RichResult(title="Intra-year region transition matrix",
                          warnings=["no within-year transitions"])
    counts = (edges.groupby(["regA_prev", "regA"]).size()
                  .unstack(fill_value=0))
    # Make square (every region as both row and col)
    regions = sorted(set(counts.index) | set(counts.columns))
    counts = counts.reindex(index=regions, columns=regions, fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    P = (counts.div(row_sums, axis=0)).fillna(0.0)

    # Stationary distribution via left-eigvec of largest eigval
    P_arr = P.to_numpy()
    if P_arr.shape[0] >= 2:
        eigvals, eigvecs = np.linalg.eig(P_arr.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        v = np.real(eigvecs[:, idx])
        v = v / v.sum() if v.sum() != 0 else v
        stationary = dict(zip(regions, [round(float(x), 4) for x in v]))
    else:
        stationary = {}

    # Off-diagonal Theil-T
    off = counts.values.copy()
    np.fill_diagonal(off, 0)
    off_total = off.sum()
    if off_total > 0:
        p = off.flatten() / off_total
        p = p[p > 0]
        theil = float((p * np.log(p * len(p))).sum())
    else:
        theil = float("nan")

    diag_share = float(np.diag(counts.values).sum() / counts.values.sum())

    return RichResult(
        title="Intra-year region transition matrix (a01 Markov)",
        summary_lines=[
            ("Transitions observed", int(counts.values.sum())),
            ("Region states", len(regions)),
            ("Diagonal share (stay-in-region)",
                f"{100*diag_share:.1f}%"),
            ("Off-diagonal Theil-T",
                round(theil, 4) if np.isfinite(theil) else "n/a"),
            ("Stationary π (regA)", stationary),
        ],
        tables=[{
            "title": "Transition probability matrix P(regA→regA'):",
            "headers": ["from \\ to"] + regions,
            "rows": [[r] + [round(float(P.loc[r, c]), 3) for c in regions]
                     for r in regions],
        }],
        interpretation=(
            f"Within-fiscal-year region-stickiness is "
            f"{100*diag_share:.1f}% (diagonal share). The off-diagonal "
            f"mass is concentrated with Theil-T = {theil:.3f} "
            "(higher = more concentrated transitions, lower = more uniform "
            "cross-region mixing). All transitions are intra-year — see "
            "OTIS_LINKAGE.md."
        ),
        payload={"diag_share": diag_share, "theil_off": theil,
                 "stationary": stationary,
                 "n_transitions": int(counts.values.sum())},
        warnings=[
            "OTIS UniqueIndividual_ID is year-locked (YYYY-XXXXX-AA); "
            "this transition matrix is intra-year only.",
        ],
    )


# ── 8. Path complexity Gini split by (year, region) ───────────────


def path_complexity_gini(df: pd.DataFrame | None = None) -> RichResult:
    """Gini of placements-per-(id × year) cells, split by (year, region).

    Extends `within_year_placement_count` by reporting per-region and
    per-year Gini coefficients, exposing where Goffmanian cycling is
    most concentrated.
    """
    df = df if df is not None else _load("b01")
    needed = {"UniqueIndividual_ID", "EndFiscalYear",
              "Region_AtTimeOfPlacement"}
    if not needed.issubset(df.columns):
        return RichResult(title="Path complexity Gini",
                          warnings=["b01 missing required cols"])
    g = df.dropna(subset=list(needed))
    cells = (g.groupby(["UniqueIndividual_ID", "EndFiscalYear",
                          "Region_AtTimeOfPlacement"])
                .size().reset_index(name="n_placements"))
    if cells.empty:
        return RichResult(title="Path complexity Gini",
                          warnings=["no usable rows"])
    overall_gini = gini(cells["n_placements"].to_numpy())
    by_yr = (cells.groupby("EndFiscalYear")["n_placements"]
                  .apply(lambda s: round(gini(s.to_numpy()), 4)))
    by_rg = (cells.groupby("Region_AtTimeOfPlacement")["n_placements"]
                  .apply(lambda s: round(gini(s.to_numpy()), 4)))
    by_yrg = (cells.groupby(
        ["EndFiscalYear", "Region_AtTimeOfPlacement"])["n_placements"]
        .apply(lambda s: round(gini(s.to_numpy()), 4))
        .reset_index(name="gini"))
    return RichResult(
        title="Path complexity Gini (b01 placement counts)",
        summary_lines=[
            ("Overall Gini", round(overall_gini, 4)),
            ("Cells", int(cells.shape[0])),
            ("Total placements", int(cells["n_placements"].sum())),
            ("Gini by fiscal year", dict(by_yr)),
            ("Gini by region", dict(by_rg)),
        ],
        tables=[{
            "title": "Gini by (year × region):",
            "headers": ["EndFiscalYear", "Region", "Gini"],
            "rows": [[int(r["EndFiscalYear"]), str(r["Region_AtTimeOfPlacement"]),
                      float(r["gini"])] for _, r in by_yrg.iterrows()],
        }],
        interpretation=(
            f"Overall placement-count Gini = {overall_gini:.3f}. "
            "Higher region- or year-specific Gini ⇒ that cell has more "
            "Goffmanian heavy-tail cycling (a few people accumulating "
            "many placements). All values are intra-year by construction."
        ),
        payload={"overall_gini": float(overall_gini),
                 "by_year": {int(k): float(v) for k, v in by_yr.items()},
                 "by_region": {str(k): float(v) for k, v in by_rg.items()}},
    )


# ── 9. Region × alert state-richness (b01) ────────────────────────


def region_alert_state_richness(df: pd.DataFrame | None = None) -> RichResult:
    """Distinct (region × alert-combo) states occupied per (id × year).

    Goffman's mortification × institutional-mobility hybrid: how many
    distinct *role-zones* (region paired with alert presentation) does
    a person traverse within one fiscal year?
    """
    df = df if df is not None else _load("b01")
    needed = {"UniqueIndividual_ID", "EndFiscalYear",
              "Region_AtTimeOfPlacement", "MentalHealth_Alert",
              "SuicideRisk_Alert", "SuicideWatch_Alert"}
    if not needed.issubset(df.columns):
        return RichResult(title="Region × alert state richness",
                          warnings=["b01 missing required cols"])
    base = df.dropna(subset=list(needed)).copy()
    a_mh = _binarise(base["MentalHealth_Alert"])
    a_sr = _binarise(base["SuicideRisk_Alert"])
    a_sw = _binarise(base["SuicideWatch_Alert"])
    base["combo"] = (a_mh * 4 + a_sr * 2 + a_sw * 1).astype(int)
    base["state"] = (base["Region_AtTimeOfPlacement"].astype(str)
                      + ":c" + base["combo"].astype(str))
    n_states = (base.groupby(["UniqueIndividual_ID", "EndFiscalYear"])
                       ["state"].nunique())
    if n_states.empty:
        return RichResult(title="Region × alert state richness",
                          warnings=["no usable rows"])
    arr = n_states.to_numpy()
    return RichResult(
        title="Region × alert state richness (Goffmanian role-zone sweep)",
        summary_lines=[
            ("Person-year cells", int(n_states.size)),
            ("Mean distinct states / cell", round(float(arr.mean()), 3)),
            ("Median", float(np.median(arr))),
            ("Max distinct states", int(arr.max())),
            ("Cells in 1 state", int((arr == 1).sum())),
            ("Cells in 2 states", int((arr == 2).sum())),
            ("Cells in 3+ states", int((arr >= 3).sum())),
            ("% cells with multi-state", f"{100*float((arr > 1).mean()):.1f}%"),
        ],
        interpretation=(
            f"{100*(arr > 1).mean():.1f}% of person-years span multiple "
            "(region × alert-combo) states within one fiscal year — a "
            "Goffmanian sweep across institutional role-zones. Possible "
            "states = 5 regions × 8 combos = 40."
        ),
        payload={"mean_states": float(arr.mean()),
                 "median_states": float(np.median(arr)),
                 "max_states": int(arr.max()),
                 "frac_multi": float((arr > 1).mean())},
    )


# ── 10. Multi-region path × demographic contingency (b01) ─────────


def regC_demog_contingency(df: pd.DataFrame | None = None) -> RichResult:
    """Multi-region path indicator (regC = number of distinct regions
    visited within FY) × Gender × Age contingency, with chi² + Cramer's V.
    """
    df = df if df is not None else _load("b01")
    needed = {"UniqueIndividual_ID", "EndFiscalYear",
              "Region_AtTimeOfPlacement", "Gender", "Age_Category"}
    if not needed.issubset(df.columns):
        return RichResult(title="Multi-region path × demographics",
                          warnings=["b01 missing required cols"])
    base = df.dropna(subset=list(needed)).copy()
    cell = (base.groupby(["UniqueIndividual_ID", "EndFiscalYear"])
                  .agg(regC=("Region_AtTimeOfPlacement", "nunique"),
                        Gender=("Gender", "first"),
                        Age_Category=("Age_Category", "first"))
                  .reset_index())
    cell["multi_region"] = (cell["regC"] >= 2).astype(int)
    if cell.empty:
        return RichResult(title="Multi-region path × demographics",
                          warnings=["no usable rows"])
    # Gender × multi_region χ² + Cramer's V
    tab_g = pd.crosstab(cell["Gender"], cell["multi_region"])
    if tab_g.shape == (2, 2) and tab_g.values.min() >= 5:
        chi_g, p_g, dof_g, _ = sps.chi2_contingency(tab_g.values)
        v_g = float(np.sqrt(chi_g / (tab_g.values.sum()
                       * (min(tab_g.shape) - 1))))
    else:
        chi_g = p_g = float("nan")
        v_g = float("nan")
    # Age × multi_region
    tab_a = pd.crosstab(cell["Age_Category"], cell["multi_region"])
    if tab_a.shape[0] >= 2 and tab_a.values.min() >= 5:
        chi_a, p_a, dof_a, _ = sps.chi2_contingency(tab_a.values)
        v_a = float(np.sqrt(chi_a / (tab_a.values.sum()
                       * (min(tab_a.shape) - 1))))
    else:
        chi_a = p_a = float("nan")
        v_a = float("nan")
    return RichResult(
        title="Multi-region path × demographics (chi² + Cramer's V)",
        summary_lines=[
            ("Person-year cells", int(cell.shape[0])),
            ("% multi-region", f"{100*float(cell['multi_region'].mean()):.1f}%"),
            ("Gender χ²", round(chi_g, 3) if np.isfinite(chi_g) else "n/a"),
            ("Gender χ² p-value",
                round(p_g, 6) if np.isfinite(p_g) else "n/a"),
            ("Gender Cramer's V",
                round(v_g, 4) if np.isfinite(v_g) else "n/a"),
            ("Age χ²", round(chi_a, 3) if np.isfinite(chi_a) else "n/a"),
            ("Age χ² p-value",
                round(p_a, 6) if np.isfinite(p_a) else "n/a"),
            ("Age Cramer's V",
                round(v_a, 4) if np.isfinite(v_a) else "n/a"),
        ],
        tables=[{
            "title": "Gender × multi-region (intra-FY):",
            "headers": ["Gender", "single-region", "multi-region"],
            "rows": [[str(g), int(tab_g.loc[g, 0]) if 0 in tab_g.columns else 0,
                       int(tab_g.loc[g, 1]) if 1 in tab_g.columns else 0]
                     for g in tab_g.index],
        }, {
            "title": "Age × multi-region (intra-FY):",
            "headers": ["Age_Category", "single-region", "multi-region"],
            "rows": [[str(a),
                       int(tab_a.loc[a, 0]) if 0 in tab_a.columns else 0,
                       int(tab_a.loc[a, 1]) if 1 in tab_a.columns else 0]
                     for a in tab_a.index],
        }],
        interpretation=(
            "Cramer's V quantifies association strength of Gender and Age "
            "with within-fiscal-year multi-region cycling. V≈0 → "
            "independence, V→1 → strong association. All measures are "
            "intra-year (OTIS IDs are year-locked)."
        ),
        payload={"frac_multi": float(cell["multi_region"].mean()),
                 "chi2_gender": float(chi_g) if np.isfinite(chi_g) else None,
                 "p_gender": float(p_g) if np.isfinite(p_g) else None,
                 "v_gender": float(v_g) if np.isfinite(v_g) else None,
                 "chi2_age": float(chi_a) if np.isfinite(chi_a) else None,
                 "p_age": float(p_a) if np.isfinite(p_a) else None,
                 "v_age": float(v_a) if np.isfinite(v_a) else None},
    )


# ── 11. Incidence-rate-ratio Poisson/NB on vm count (b01) ─────────


def irr_glmm_vm(df: pd.DataFrame | None = None) -> RichResult:
    """Poisson + Negative-Binomial GLM IRR table for vm ~ T_high_ac
    plus demographic covariates, on the (UniqueIndividual_ID, FY) cell.

    A simplified implementation of the OTIS-RC `glmmTMB` pipeline:
    no random effect (statsmodels can't easily do mixed NB on this
    scale). Provides Poisson and NB2 IRR side by side as a robustness
    pair.
    """
    df = df if df is not None else _load("b01")
    try:
        from .otis_causal import make_pair_alert_to_volatility_ruhela
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError as e:
        return RichResult(title="IRR Poisson/NB GLM (vm)",
                          warnings=[f"missing dep: {e}"])
    py, T, Y, _ = make_pair_alert_to_volatility_ruhela(df)
    if py.empty or py[Y].sum() == 0:
        return RichResult(title="IRR Poisson/NB GLM (vm)",
                          warnings=["empty cell-table or zero outcome"])
    py = py.copy()
    py["yr"] = py["EndFiscalYear"].astype("category")
    py["sg"] = py["Gender"].astype("category")
    py["ag"] = py["Age_Category"].astype("category")
    formula = f"{Y} ~ {T} + C(yr) + C(sg) + C(ag)"

    out_rows = []
    for family_label, family in (
        ("Poisson", sm.families.Poisson()),
        ("NegBin2", sm.families.NegativeBinomial(alpha=1.0)),
    ):
        try:
            model = smf.glm(formula, data=py, family=family).fit()
            beta = float(model.params.get(T, np.nan))
            se = float(model.bse.get(T, np.nan))
            irr = float(np.exp(beta)) if np.isfinite(beta) else float("nan")
            ci_lo = float(np.exp(beta - 1.96 * se)) if np.isfinite(se) else float("nan")
            ci_hi = float(np.exp(beta + 1.96 * se)) if np.isfinite(se) else float("nan")
            pval = float(model.pvalues.get(T, np.nan))
            out_rows.append([family_label, round(irr, 4),
                              f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                              round(pval, 6),
                              round(float(model.aic), 2)])
        except Exception as e:  # noqa: BLE001
            out_rows.append([family_label, "fit failed",
                              str(type(e).__name__), "—", "—"])
    return RichResult(
        title="IRR Poisson/NB GLM — vm ~ T_high_ac + demog",
        summary_lines=[
            ("Cells (id × year)", int(py.shape[0])),
            ("Mean vm / cell", round(float(py[Y].mean()), 3)),
            ("Treatment T_high_ac (ac≥2) prevalence",
                f"{100*float(py[T].mean()):.1f}%"),
        ],
        tables=[{
            "title": f"IRR for {T} on {Y} (covariate-adjusted):",
            "headers": ["Family", "IRR", "95% CI", "p-value", "AIC"],
            "rows": out_rows,
        }],
        interpretation=(
            "IRR ≈ exp(β) on the alert-complexity treatment. "
            "Poisson assumes equidispersion; NB2 relaxes that. "
            "Concordance between the two is a robustness signal. "
            "All inference is on intra-year (id × FY) cells."
        ),
        payload={"n_cells": int(py.shape[0]),
                 "mean_vm": float(py[Y].mean()),
                 "irr_results": out_rows},
        warnings=[
            "No random-effect / cluster-robust SE in this implementation. "
            "For paper-grade SEs, use morie.otis_causal.otis_irm_dml.",
        ],
    )


def analyze_all(out_dir: Path | None = None) -> dict[str, RichResult]:
    out_dir = out_dir or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    for name, fn in [
        ("repeat_placement_concentration", repeat_placement_concentration),
        ("within_year_placement_count", within_year_placement_count),
        ("within_year_region_diversity", within_year_region_diversity),
        ("mortification_cooccurrence", mortification_cooccurrence),
        ("disciplinary_medical_overlap", disciplinary_medical_overlap),
        ("embedding_distribution", embedding_distribution),
        ("intra_year_transition_matrix", intra_year_transition_matrix),
        ("path_complexity_gini", path_complexity_gini),
        ("region_alert_state_richness", region_alert_state_richness),
        ("regC_demog_contingency", regC_demog_contingency),
        ("irr_glmm_vm", irr_glmm_vm),
    ]:
        try:
            r = fn()
            results[name] = r
            (out_dir / f"churn_{name}.txt").write_text(str(r))
            (out_dir / f"churn_{name}.json").write_text(
                json.dumps(r.payload, indent=2, default=str,
                           ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            results[name] = RichResult(
                title=f"churn {name} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
