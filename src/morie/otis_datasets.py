"""morie.otis_datasets -- registry + loader for all 29 OTIS CSVs.

The Ontario Tracking Information System (OTIS) is the open-data feed
for Ontario's adult correctional system, published by the Ministry
of the Solicitor General's Service Management and Oversight Branch
under the Jahn settlement framework.  The 29 datasets in the
official A01RCDD release fall into 4 series:

    a-series (1 dataset)    -- Restrictive confinement detailed
                              (the canonical per-day record file used
                              in Ruhela 2026 OTIS-RC)
    b-series (9 datasets)   -- Segregation placement records
    c-series (12 datasets)  -- Individuals in segregation & restrictive
                              confinement
    d-series (7 datasets)   -- Deaths in custody

CSV path convention:
    data/datasets/OTIS/<id>_<short_label>.csv

Where <id> is the canonical dataset name from the official Ontario
data dictionary (a01, b01, c03, d05, …).

Public API:
    DATASET_REGISTRY        -- dict mapping id -> metadata
    load_otis_dataset(id)   -- returns DataFrame
    list_otis_datasets()    -- list of all (id, description, n_cols)
    download_otis_dataset(id, target_dir=None)
                            -- fetch a fresh copy from data.ontario.ca
                              via the CKAN action API
    download_all_otis(target_dir=None)
                            -- fetch every file in the package
    OTIS_DATA_DIR           -- Path
    OTIS_CKAN_PACKAGE       -- package id on data.ontario.ca

References
----------
Ministry of the Solicitor General. Data on Inmates in Ontario.
  https://data.ontario.ca/dataset/data-on-inmates-in-ontario
Ministry of the Solicitor General. Restrictive Confinement,
  Segregation and Deaths in Custody Data Dictionary, 2025-11-03.
  https://data.ontario.ca/dataset/data-on-inmates-in-ontario/resource/d83fe893-9634-4794-a0c1-c17bf619a95a
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


def _otis_cache_dir() -> Path:
    """Per-user cache directory for downloaded OTIS CSVs."""
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root) if root else Path.home() / ".cache"
    return base / "morie" / "otis"


def _resolve_otis_data_dir() -> Path:
    """Locate the directory holding the OTIS CSVs.

    Order: ``$MORIE_OTIS_DIR`` (explicit override); then an ancestor
    ``data/datasets/OTIS`` of this file (the source-tree checkout);
    then ``./data/datasets/OTIS``; then the per-user cache dir -- which
    is also where the CKAN downloader writes, so a fresh install
    resolves there and load_otis_dataset's auto-download populates it.

    The old ``parents[5]`` hard-coded depth was wrong from both an
    installed wheel and the source tree.
    """
    env = os.environ.get("MORIE_OTIS_DIR")
    if env:
        return Path(env).expanduser()
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "datasets" / "OTIS"
        if cand.is_dir():
            return cand
    cwd = Path.cwd() / "data" / "datasets" / "OTIS"
    if cwd.is_dir():
        return cwd
    return _otis_cache_dir()


OTIS_DATA_DIR = _resolve_otis_data_dir()


@dataclass(frozen=True, slots=True)
class OtisDataset:
    """One OTIS dataset's metadata."""
    id: str                  # b01, c03, d05, ...
    series: str              # 'b', 'c', or 'd'
    csv_filename: str        # b01_segregation_detailed_dataset.csv
    description: str
    columns: tuple[str, ...]  # canonical column names
    panel: bool              # True if person-level, False if aggregate
    primary_metric: str       # the headline value column

    @property
    def csv_path(self) -> Path:
        return OTIS_DATA_DIR / self.csv_filename


DATASET_REGISTRY: dict[str, OtisDataset] = {}


def _r(d: OtisDataset) -> None:
    DATASET_REGISTRY[d.id] = d


# ── a-series: Restrictive confinement detailed (Ruhela 2026 primary) ──
_r(OtisDataset(
    id="a01", series="a",
    csv_filename="a01_restrictive_confinement_detailed_dataset.csv",
    description=("Restrictive confinement -- person-level detail "
                  "(each row = one DAY in restrictive confinement). "
                  "The canonical RC dataset used in Ruhela's OTIS-RC "
                  "research (notez1a.qmd, res_pool, res_by_year, "
                  "res_all)."),
    columns=("EndFiscalYear", "UniqueIndividual_ID",
              "Region_AtTimeOfPlacement", "Region_MostRecentPlacement",
              "Gender", "Age_Category",
              "MentalHealth_Alert", "SuicideRisk_Alert",
              "SuicideWatch_Alert", "Number_Of_Placements"),
    panel=True, primary_metric="Number_Of_Placements",
))


# ── b-series: Segregation placements ────────────────────────────────
_r(OtisDataset(
    id="b01", series="b",
    csv_filename="b01_segregation_detailed_dataset.csv",
    description="Segregation placements -- person-level detail (each row = one placement)",
    columns=("EndFiscalYear", "UniqueIndividual_ID", "Gender",
             "Region_AtTimeOfPlacement", "Region_MostRecentPlacement",
             "Age_Category", "NumberConsecutiveDays_Segregation",
             "SegReason_SecurityOfInstitution_SafetyOfOthers",
             "SegReason_InmateNeedsProtection",
             "SegReason_InmateNeedsProtection_Medical",
             "SegReason_SecurityOfInstitution_SafetyOfOthers_Medical",
             "SegReason_Disciplinary_Segregation",
             "SegReason_InmateRefuseSearch_Scan",
             "SegReason_Other",
             "MentalHealth_Alert", "SuicideRisk_Alert",
             "SuicideWatch_Alert", "Number_Of_Placements"),
    panel=True, primary_metric="Number_Of_Placements",
))
_r(OtisDataset(
    id="b02", series="b",
    csv_filename="b02_segregation_detailed_total_days.csv",
    description="Segregation total days per individual per fiscal year",
    columns=("EndFiscalYear", "UniqueIndividual_ID", "Gender",
             "Region_MostRecentPlacement", "Age_Category",
             "TotalAggregatedDays_Segregation"),
    panel=True, primary_metric="TotalAggregatedDays_Segregation",
))
_r(OtisDataset(
    id="b03", series="b",
    csv_filename="b03_segregation_placements_alerts_and_hold_flags_by_institution.csv",
    description="Segregation placements by alert/hold flag × institution",
    columns=("EndFiscalYear", "Region_AtTimeOfPlacement",
             "Institution_AtTimeOfPlacement", "Alert_Type",
             "Alert_Presence", "Number_SegregationPlacements"),
    panel=False, primary_metric="Number_SegregationPlacements",
))
_r(OtisDataset(
    id="b04", series="b",
    csv_filename="b04_segregation_placements_consecutive_durations_by_region.csv",
    description="Segregation placement durations (max/median/mode) by region & gender",
    columns=("EndFiscalYear", "Region_AtTimeOfPlacement",
             "Gender", "Measure", "NumberConsecutiveDays_Segregation"),
    panel=False, primary_metric="NumberConsecutiveDays_Segregation",
))
_r(OtisDataset(
    id="b05", series="b",
    csv_filename="b05_segregation_placements_consecutive_lengths.csv",
    description="Segregation placement count by binned duration",
    columns=("EndFiscalYear", "Consecutive_Duration",
             "Number_SegregationPlacements"),
    panel=False, primary_metric="Number_SegregationPlacements",
))
_r(OtisDataset(
    id="b06", series="b",
    csv_filename="b06_segregation_placements_reason_for_placement_by_institution.csv",
    description="Segregation placements by reason × institution × gender",
    columns=("EndFiscalYear", "Region_AtTimeOfPlacement",
             "Institution_AtTimeOfPlacement", "Gender", "Reason",
             "Number_SegregationPlacements"),
    panel=False, primary_metric="Number_SegregationPlacements",
))
_r(OtisDataset(
    id="b07", series="b",
    csv_filename="b07_segregation_placements_alerts_and_hold_flags_by_gender.csv",
    description="Segregation placements with/without alert × gender",
    columns=("EndFiscalYear", "Alert_Type", "Gender",
             "Number_Segregation_Placements_Without_Alert",
             "Number_Segregation_Placements_With_Alert"),
    panel=False, primary_metric="Number_Segregation_Placements_With_Alert",
))
_r(OtisDataset(
    id="b08", series="b",
    csv_filename="b08_segregation_placements_consecutive_durations_by_institution.csv",
    description="Segregation placement durations (median/mode) by institution & gender",
    columns=("EndFiscalYear", "Region_AtTimeOfPlacement",
             "Institution_AtTimeOfPlacement", "Gender", "Measure",
             "NumberConsecutiveDays_Segregation"),
    panel=False, primary_metric="NumberConsecutiveDays_Segregation",
))
_r(OtisDataset(
    id="b09", series="b",
    csv_filename="b09_individuals_in_segregation_number_of_times_in_segregation.csv",
    description="Individuals by number of segregation placements × gender",
    columns=("EndFiscalYear", "NumberPlacements_Segregation",
             "Gender", "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_Segregation",
))

# ── c-series: Individuals in segregation & restrictive confinement ──
_r(OtisDataset(
    id="c01", series="c",
    csv_filename="c01_individuals_in_segregation_and_restrictive_confinement_total_individuals.csv",
    description="Total unique individuals in custody/RC/segregation × gender",
    columns=("EndFiscalYear", "Gender",
             "NumberIndividuals_InCustody",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c02", series="c",
    csv_filename="c02_individuals_in_segregation_and_restrictive_confinement_by_institution.csv",
    description="Individuals in RC/segregation by institution × region × gender",
    columns=("EndFiscalYear", "Region_MostRecentPlacement",
             "Institution_MostRecentPlacement", "Gender",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c03", series="c",
    csv_filename="c03_individuals_in_segregation_and_restrictive_confinement_race_by_gender.csv",
    description="Individuals in custody/RC/seg × race × gender",
    columns=("EndFiscalYear", "Race", "Gender",
             "NumberIndividuals_InCustody",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c04", series="c",
    csv_filename="c04_individuals_in_segregation_and_restrictive_confinement_race_by_region.csv",
    description="Individuals in RC/seg × race × region",
    columns=("EndFiscalYear", "Region_MostRecentPlacement",
             "Race",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c05", series="c",
    csv_filename="c05_individuals_in_segregation_and_restrictive_confinement_religion_by_region.csv",
    description="Individuals in RC/seg × religion × region",
    columns=("EndFiscalYear", "Region_MostRecentPlacement",
             "Religion",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c06", series="c",
    csv_filename="c06_individuals_in_segregation_and_restrictive_confinement_age_category_by_region.csv",
    description="Individuals in RC/seg × age category × region",
    columns=("EndFiscalYear", "Region_MostRecentPlacement",
             "Age_Category",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c07", series="c",
    csv_filename="c07_individuals_in_segregation_and_restrictive_confinement_alerts_and_hold_flags.csv",
    description="Individuals in custody/RC/seg × alert type × gender",
    columns=("EndFiscalYear", "Alert_Type", "Gender",
             "NumberIndividuals_InCustody",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c08", series="c",
    csv_filename="c08_individuals_in_segregation_and_restrictive_confinement_religion_by_gender.csv",
    description="Individuals in custody/RC/seg × religion × gender",
    columns=("EndFiscalYear", "Religion", "Gender",
             "NumberIndividuals_InCustody",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c09", series="c",
    csv_filename="c09_individuals_in_segregation_and_restrictive_confinement_age_category_by_gender.csv",
    description="Individuals in custody/RC/seg × age category × gender",
    columns=("EndFiscalYear", "Age_Category", "Gender",
             "NumberIndividuals_InCustody",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c10", series="c",
    csv_filename="c10_segregation_and_restrictive_confinement_aggregate_durations_by_institution.csv",
    description="RC/seg aggregate durations (max/median/mode) by institution",
    columns=("EndFiscalYear", "Region_MostRecentPlacement",
             "Institution_MostRecentPlacement", "Gender", "Measure",
             "TotalAggregatedDays_RestrictiveConfinement",
             "TotalAggregatedDays_Segregation"),
    panel=False, primary_metric="TotalAggregatedDays_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c11", series="c",
    csv_filename="c11_individuals_in_segregation_and_restrictive_confinement_aggregate_lengths.csv",
    description="Individuals by binned aggregate duration",
    columns=("EndFiscalYear", "Aggregate_Duration",
             "NumberIndividuals_RestrictiveConfinement",
             "NumberIndividuals_Segregation"),
    panel=False, primary_metric="NumberIndividuals_RestrictiveConfinement",
))
_r(OtisDataset(
    id="c12", series="c",
    csv_filename="c12_segregation_and_restrictive_confinement_aggregate_durations_by_region.csv",
    description="RC/seg aggregate durations (max/median/mode) by region & gender",
    columns=("EndFiscalYear", "Region_MostRecentPlacement",
             "Gender", "Measure",
             "TotalAggregatedDays_RestrictiveConfinement",
             "TotalAggregatedDays_Segregation"),
    panel=False, primary_metric="TotalAggregatedDays_RestrictiveConfinement",
))

# ── d-series: Deaths in custody ─────────────────────────────────────
_r(OtisDataset(
    id="d01", series="d",
    csv_filename="d01_deaths_in_custody_detailed_dataset.csv",
    description="Custodial deaths -- person-level detail (calendar year)",
    columns=("Year", "UniqueIndividual_ID",
             "Region_AtTimeOfDeath", "HousingUnit_Type",
             "MedicalCauseofDeath", "MeansofDeath"),
    panel=True, primary_metric="UniqueIndividual_ID",
))
_r(OtisDataset(
    id="d02", series="d",
    csv_filename="d02_deaths_in_custody_gender.csv",
    description="Custodial deaths × gender",
    columns=("Year", "Gender", "Number_CustodialDeaths"),
    panel=False, primary_metric="Number_CustodialDeaths",
))
_r(OtisDataset(
    id="d03", series="d",
    csv_filename="d03_deaths_in_custody_race.csv",
    description="Custodial deaths × race",
    columns=("Year", "Race", "Number_CustodialDeaths"),
    panel=False, primary_metric="Number_CustodialDeaths",
))
_r(OtisDataset(
    id="d04", series="d",
    csv_filename="d04_deaths_in_custody_religion.csv",
    description="Custodial deaths × religion",
    columns=("Year", "Religion", "Number_CustodialDeaths"),
    panel=False, primary_metric="Number_CustodialDeaths",
))
_r(OtisDataset(
    id="d05", series="d",
    csv_filename="d05_deaths_in_custody_age_category.csv",
    description="Custodial deaths × age category",
    columns=("Year", "Age_Category", "Number_CustodialDeaths"),
    panel=False, primary_metric="Number_CustodialDeaths",
))
_r(OtisDataset(
    id="d06", series="d",
    csv_filename="d06_deaths_in_custody_cause_of_death_by_alert_by_institution.csv",
    description="Custodial deaths × alert × medical cause",
    columns=("Year", "Alert_Type", "MedicalCauseOfDeath",
             "Number_CustodialDeaths"),
    panel=False, primary_metric="Number_CustodialDeaths",
))
_r(OtisDataset(
    id="d07", series="d",
    csv_filename="d07_deaths_in_custody_alerts_by_housing_unit.csv",
    description="Custodial deaths × alert × housing unit type",
    columns=("Year", "Alert_Type", "Housing_Type",
             "Number_CustodialDeaths"),
    panel=False, primary_metric="Number_CustodialDeaths",
))


def load_otis_dataset(dataset_id: str,
                      data_dir: Path | str | None = None,
                      *, download: bool = True) -> pd.DataFrame:
    """Load one OTIS dataset by id (b01, c03, d05, …).

    Returns a pandas DataFrame with the columns documented in
    `DATASET_REGISTRY[dataset_id].columns`.

    The OTIS CSVs are Ontario open data. If the file is not present
    locally and ``download`` is true (default), it is fetched once from
    the Ontario CKAN portal into the per-user cache and reused
    thereafter. Set ``download=False`` or ``$MORIE_OTIS_NO_DOWNLOAD`` to
    require a local copy instead.
    """
    dataset_id = dataset_id.lower()
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(
            f"unknown OTIS dataset id {dataset_id!r}. "
            f"valid ids: {sorted(DATASET_REGISTRY.keys())}"
        )
    meta = DATASET_REGISTRY[dataset_id]
    base = Path(data_dir) if data_dir else OTIS_DATA_DIR
    p = base / meta.csv_filename
    if not p.exists():
        if download and not os.environ.get("MORIE_OTIS_NO_DOWNLOAD"):
            # Ontario open data -- fetch once into the per-user cache.
            p = download_otis_dataset(dataset_id, _otis_cache_dir())
        else:
            raise FileNotFoundError(
                f"OTIS dataset {dataset_id} not found at {p}. The OTIS "
                f"CSVs are Ontario open data: call "
                f"morie.otis_datasets.download_all_otis(), or set "
                f"$MORIE_OTIS_DIR to a directory holding the 29 CSVs."
            )
    return pd.read_csv(p)


def list_otis_datasets() -> list[tuple[str, str, int]]:
    """List all 29 OTIS datasets as (id, description, n_columns)."""
    return [
        (d.id, d.description, len(d.columns))
        for d in sorted(DATASET_REGISTRY.values(), key=lambda x: x.id)
    ]


# ── CKAN downloader ────────────────────────────────────────────────


OTIS_CKAN_PACKAGE = "data-on-inmates-in-ontario"
OTIS_CKAN_BASE = "https://data.ontario.ca/api/3/action"


def _ckan_resource_index() -> dict[str, str]:
    """Return ``{file_basename: csv_url}`` for every resource in the
    Ontario package, fetched fresh from the CKAN ``package_show`` API.

    Resource ``name`` is the file basename (e.g.\\
    ``a01_restrictive_confinement_detailed_dataset``); we match by
    case-insensitive prefix on the registry's ``csv_filename``
    (without ``.csv``).
    """
    import json
    import urllib.request

    url = f"{OTIS_CKAN_BASE}/package_show?id={OTIS_CKAN_PACKAGE}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not body.get("success"):
        raise RuntimeError(f"CKAN package_show failed: {body}")
    out: dict[str, str] = {}
    for r in body["result"]["resources"]:
        name = (r.get("name") or "").lower()
        u = r.get("url") or ""
        if name and u and u.lower().endswith(".csv"):
            out[name] = u
    return out


def download_otis_dataset(dataset_id: str,
                            target_dir: str | Path | None = None,
                            *, overwrite: bool = False) -> Path:
    """Fetch one OTIS CSV from CKAN and write to ``target_dir``.

    Returns the local path to the downloaded file.  No-op if the
    file already exists and ``overwrite=False``.
    """
    import urllib.request

    dataset_id = dataset_id.lower()
    if dataset_id not in DATASET_REGISTRY:
        raise KeyError(f"unknown OTIS dataset id {dataset_id!r}")
    meta = DATASET_REGISTRY[dataset_id]
    base = Path(target_dir) if target_dir else OTIS_DATA_DIR
    base.mkdir(parents=True, exist_ok=True)
    out_path = base / meta.csv_filename
    if out_path.exists() and not overwrite:
        return out_path

    idx = _ckan_resource_index()
    # Build token set from local filename -- strip leading "<id>_",
    # drop common boilerplate words, lowercase.  Then match against
    # CKAN names (which use spaces / en-dashes / mixed case).
    import re
    needle_raw = meta.csv_filename.removesuffix(".csv").lower()
    needle_tokens = set(t for t in re.split(r"[^a-z0-9]+", needle_raw)
                          if t and t not in {"dataset", meta.id, "and"})
    url = idx.get(needle_raw)
    best_score, best_url = 0, None
    if url is None:
        for name, u in idx.items():
            name_tokens = set(t for t in re.split(r"[^a-z0-9]+",
                                                    name.lower())
                                if t and t not in {"dataset", "and"})
            score = len(needle_tokens & name_tokens)
            if score > best_score:
                best_score, best_url = score, u
        url = best_url
    if url is None or best_score < 2:
        raise RuntimeError(
            f"no CKAN resource matched {meta.csv_filename} "
            f"(best_score={best_score}); "
            f"available: {sorted(idx.keys())[:6]}…")
    with urllib.request.urlopen(url, timeout=120) as resp:
        out_path.write_bytes(resp.read())
    return out_path


def download_all_otis(target_dir: str | Path | None = None,
                       *, overwrite: bool = False,
                       skip_missing: bool = True) -> dict[str, Path]:
    """Fetch every CSV in the OTIS registry from CKAN.

    Returns ``{dataset_id: local_path}`` for successfully downloaded
    files.  When ``skip_missing=True`` (default), files that aren't
    found on CKAN are silently skipped; otherwise the function raises.
    """
    out: dict[str, Path] = {}
    for did in DATASET_REGISTRY:
        try:
            out[did] = download_otis_dataset(did, target_dir,
                                              overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            if skip_missing:
                continue
            raise
    return out
