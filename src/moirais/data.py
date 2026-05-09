import json
import logging
import os
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

_SAFE_TABLE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_table_name(name: str) -> str:
    """Validate a SQLite table name contains only safe characters."""
    if not _SAFE_TABLE_RE.match(name):
        raise ValueError(f"Unsafe table name: {name!r}")
    return name


from .cpads import (
    CPADS_REQUIRED_VARIABLES,
    canonicalize_cpads_frame,
    cpads_contract,
    has_raw_cpads_columns,
    infer_file_format,
    validate_cpads_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_CKAN_API_BASE = "https://open.canada.ca/data/en/api/3/action/datastore_search"
DEFAULT_CACHE_DB = "data/cache/moirais.db"


def _project_root() -> Path:
    """Return the project root (dev/sphinx/project/)."""
    # __file__ = libexec/config/tools/py-package/moirais/data.py
    # parents: [0]=moirais [1]=py-package [2]=tools [3]=config [4]=libexec [5]=project
    return Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# CKAN dataset catalogue — Canadian public health open data
# ---------------------------------------------------------------------------

CKAN_DATASETS: dict[str, dict[str, str]] = {
    "cpads": {
        "name": "CPADS 2021-2022 PUMF",
        "package_id": "736fa9b2-62e4-4e31-aea4-51869605b363",
        "resource_id": "d2639429-c304-45a6-90b3-770562f4d46d",
        "metadata_url": "https://open.canada.ca/data/api/action/package_show?id=736fa9b2-62e4-4e31-aea4-51869605b363",
    },
    "csads": {
        "name": "CSADS",
        "package_id": "1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
        "resource_id": "f6761337-47e9-455a-a3c4-ea8516aa634f",  # 2021-2022 CSTADS PUMF CSV
        "metadata_url": "https://open.canada.ca/data/api/action/package_show?id=1f15ca45-8bfd-4f9c-9ec6-2c0c440e69c2",
    },
    "csus": {
        "name": "CSUS",
        "package_id": "65e2d45e-efc6-4c29-9a9b-db59bc96aa0e",
        "resource_id": "c2c1795b-4501-49ba-9dd1-5b8360cc3b2e",  # 2023 CSUS PUMF CSV (verified via package_show 2026-04-18)
        "metadata_url": "https://open.canada.ca/data/api/action/package_show?id=65e2d45e-efc6-4c29-9a9b-db59bc96aa0e",
    },
}


# ---------------------------------------------------------------------------
# Full dataset catalog — every file in data/datasets/
# ---------------------------------------------------------------------------

DATASET_CATALOG: dict[str, dict] = {
    # ── OpenCanada (oc) PUMF microdata ────────────────────────
    "ocp21": {
        "name": "CPADS 2021-2022 PUMF",
        "source": "oc",
        "survey": "cpads",
        "year": "2021-2022",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
        "table_name": "ocp21",
        "ckan_resource_id": "d2639429-c304-45a6-90b3-770562f4d46d",
    },
    "occ22": {
        "name": "CCS 2018-2022 PUMF",
        "source": "oc",
        "survey": "ccs",
        "year": "2018-2022",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CCS/2018-2022/ccs_pumf_2018to2022_final.csv",
        "table_name": "occ22",
        "ckan_resource_id": "",
    },
    "occ23": {
        "name": "CCS 2023 PUMF",
        "source": "oc",
        "survey": "ccs",
        "year": "2023",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CCS/2023/ccs_2023_pumf.csv",
        "table_name": "occ23",
        "ckan_resource_id": "",
    },
    "occ24": {
        "name": "CCS 2024 PUMF",
        "source": "oc",
        "survey": "ccs",
        "year": "2024",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CCS/2024/ccs_pumf_2024-002.csv",
        "table_name": "occ24",
        "ckan_resource_id": "",
    },
    "ocs22mf": {
        "name": "CSADS 2021-2022 PUMF",
        "source": "oc",
        "survey": "csads",
        "year": "2021-2022",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSADS/2021-2022/csads202122pumf.csv",
        "table_name": "ocs22mf",
        "ckan_resource_id": "f6761337-47e9-455a-a3c4-ea8516aa634f",
    },
    "ocs22bt": {
        "name": "CSADS 2021-2022 Bootstrap",
        "source": "oc",
        "survey": "csads",
        "year": "2021-2022",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSADS/2021-2022/csads202122bootstrap.csv",
        "table_name": "ocs22bt",
        "ckan_resource_id": "ebdc36e1-910d-4685-81a3-6acfe44729bc",
    },
    "ocs24mf": {
        "name": "CSADS 2023-2024 PUMF",
        "source": "oc",
        "survey": "csads",
        "year": "2023-2024",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSADS/2023-2024/csads202324pumf.csv",
        "table_name": "ocs24mf",
        "ckan_resource_id": "81a3adf0-61d0-4691-afba-588fa5f563da",
    },
    "ocs24bt": {
        "name": "CSADS 2023-2024 Bootstrap",
        "source": "oc",
        "survey": "csads",
        "year": "2023-2024",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSADS/2023-2024/csads202324bootstrap.csv",
        "table_name": "ocs24bt",
        "ckan_resource_id": "",
    },
    "cu20mf": {
        "name": "CSUS 2019-2020 PUMF",
        "source": "oc",
        "survey": "csus",
        "year": "2019-2020",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSUS/2019-2020/CADS201920pumf.csv",
        "table_name": "cu20mf",
        "ckan_resource_id": "",
    },
    "cu20bt": {
        "name": "CSUS 2019-2020 Bootstrap",
        "source": "oc",
        "survey": "csus",
        "year": "2019-2020",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSUS/2019-2020/CADS201920bsw.csv",
        "table_name": "cu20bt",
        "ckan_resource_id": "",
    },
    "cu23mf": {
        "name": "CSUS 2023 PUMF",
        "source": "oc",
        "survey": "csus",
        "year": "2023",
        "format": "csv",
        "type": "pumf",
        "large_file": False,
        "local_path": "data/datasets/oc/CSUS/2023/csus2023_pumf_final.csv",
        "table_name": "cu23mf",
        "ckan_resource_id": "",
    },
    "cu23bt": {
        "name": "CSUS 2023 Bootstrap",
        "source": "oc",
        "survey": "csus",
        "year": "2023",
        "format": "csv",
        "type": "bootstrap",
        "large_file": True,
        "local_path": "data/datasets/oc/CSUS/2023/csus2023_pumf_bwt.csv",
        "table_name": "cu23bt",
        "ckan_resource_id": "",
    },
    # ── HealthInfobase (hib) aggregate ────────────────────────
    "hibp": {
        "name": "CPADS Aggregate",
        "source": "hib",
        "survey": "cpads",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CPADS/CPADS.csv",
        "table_name": "hibp",
        "ckan_resource_id": "",
    },
    "hibsa": {
        "name": "CSADS Provinces",
        "source": "hib",
        "survey": "csads",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSADS/provinces.csv",
        "table_name": "hibsa",
        "ckan_resource_id": "",
    },
    "hibsb": {
        "name": "CSADS Trends",
        "source": "hib",
        "survey": "csads",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSADS/trends.csv",
        "table_name": "hibsb",
        "ckan_resource_id": "",
    },
    "hibua": {
        "name": "CSUS Alcohol",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Alcohol.csv",
        "table_name": "hibua",
        "ckan_resource_id": "",
    },
    "hibub": {
        "name": "CSUS Cannabis",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Cannabis.csv",
        "table_name": "hibub",
        "ckan_resource_id": "",
    },
    "hibuc": {
        "name": "CSUS Smoking & Vaping",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Cigarette smoking and vaping.csv",
        "table_name": "hibuc",
        "ckan_resource_id": "",
    },
    "hibud": {
        "name": "CSUS Illegal Substances",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Illegal substances.csv",
        "table_name": "hibud",
        "ckan_resource_id": "",
    },
    "hibue": {
        "name": "CSUS Opioids",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Opioids.csv",
        "table_name": "hibue",
        "ckan_resource_id": "",
    },
    "hibuf": {
        "name": "CSUS OTC Products",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Over the counter products.csv",
        "table_name": "hibuf",
        "ckan_resource_id": "",
    },
    "hibug": {
        "name": "CSUS Polysubstance",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Polysubstance.csv",
        "table_name": "hibug",
        "ckan_resource_id": "",
    },
    "hibuh": {
        "name": "CSUS Sedatives",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Sedatives.csv",
        "table_name": "hibuh",
        "ckan_resource_id": "",
    },
    "hibui": {
        "name": "CSUS Stimulants",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Stimulants.csv",
        "table_name": "hibui",
        "ckan_resource_id": "",
    },
    "hibuj": {
        "name": "CSUS Substance Use Harms",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Substance use harms.csv",
        "table_name": "hibuj",
        "ckan_resource_id": "",
    },
    "hibuk": {
        "name": "CSUS Treatment",
        "source": "hib",
        "survey": "csus",
        "year": "",
        "format": "csv",
        "type": "aggregate",
        "large_file": False,
        "local_path": "data/datasets/hib/CSUS/Treatment.csv",
        "table_name": "hibuk",
        "ckan_resource_id": "",
    },
    # ── CIHI indicator library ────────────────────────────────────────
    # ── CIHI (cihi) indicator library ───────────────────────────
    "cihidt": {
        "name": "CIHI All Indicators",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": "data/datasets/cihi/indicator-library-all-indicator-data-en.xlsx",
        "table_name": "cihidt",
        "ckan_resource_id": "",
    },
    "cihi820a": {
        "name": "CIHI 820: Substance Use Harm",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": "data/datasets/cihi/820/820-hospital-stays-for-harm-caused-by-substance-use-data-table-en.xlsx",
        "table_name": "cihi820a",
        "ckan_resource_id": "",
    },
    "cihi820b": {
        "name": "CIHI 820: Substance Use Breakdown 2024-2025",
        "source": "cihi",
        "survey": "indicators",
        "year": "2024-2025",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": (
            "data/datasets/cihi/820/"
            "820-hospital-stays-harm-due-to-substance-use-breakdown-2024-2025-data-tables-en-additional.xlsx"
        ),
        "table_name": "cihi820b",
        "ckan_resource_id": "",
    },
    "cihi849": {
        "name": "CIHI 849: Alcohol Use Harm",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": "data/datasets/cihi/849/849-hospital-stays-for-harm-caused-by-alcohol-use-data-table-en.xlsx",
        "table_name": "cihi849",
        "ckan_resource_id": "",
    },
    "cihi885a": {
        "name": "CIHI 885: Youth Services",
        "source": "cihi",
        "survey": "indicators",
        "year": "",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": (
            "data/datasets/cihi/885/"
            "885-youth-age-12-to-25-who-accessed-integrated-youth-services-for-mental-health"
            "-substance-use-and-well-being-support-data-table-en.xlsx"
        ),
        "table_name": "cihi885a",
        "ckan_resource_id": "",
    },
    "cihi885b": {
        "name": "CIHI 885: Youth Sites 2024-2025",
        "source": "cihi",
        "survey": "indicators",
        "year": "2024-2025",
        "format": "xlsx",
        "type": "indicator",
        "large_file": False,
        "local_path": (
            "data/datasets/cihi/885/"
            "885-number-integrated-youth-services-sites-2024-2025-data-tables-en-additonal.xlsx"
        ),
        "table_name": "cihi885b",
        "ckan_resource_id": "",
    },
    # ── VSR Research Data ─────────────────────────────────────
    "mapq": {
        "name": "MAPQ: Modified Attitudes on Psychedelics Questionnaire",
        "source": "vsr",
        "survey": "mapq",
        "year": "2026",
        "format": "xlsx",
        "type": "psychometric",
        "large_file": False,
        "local_path": "data/datasets/vsr/TKARONTOMAPQ.xlsx",
        "table_name": "mapq",
        "ckan_resource_id": "",
        "sheets": {
            "MAPQII": "20-item Likert scale (EE/EA/UA/ER, 4 subscales)",
            "MAPQ + KS + KnAcqS": "MAPQ subscales + Knowledge Scale + demographics",
            "recode": "Original recoded APQ items",
            "KS test items": "Knowledge Scale drug identification items",
        },
    },
    "otis": {
        "name": "OTIS: Ontario Restrictive Confinement 2023-2025",
        "source": "vsr",
        "survey": "otis",
        "year": "2023-2025",
        "format": "rdata",
        "type": "correctional",
        "large_file": False,
        "local_path": "data/cache/correctional_stats_report_environment1b.RData",
        "table_name": "otis",
        "ckan_resource_id": "",
    },
    "otisexp": {
        "name": "OTIS Expanded (1.9M placement records)",
        "source": "vsr",
        "survey": "otis",
        "year": "2023-2025",
        "format": "rds",
        "type": "correctional",
        "large_file": True,
        "local_path": "data/cache/dt_expanded.rds",
        "table_name": "otisexp",
        "ckan_resource_id": "",
    },
    "otisfin": {
        "name": "OTIS Complete Analysis Environment (239 objects)",
        "source": "vsr",
        "survey": "otis",
        "year": "2023-2025",
        "format": "rdata",
        "type": "correctional",
        "large_file": True,
        "local_path": "data/cache/finne_env.RData",
        "table_name": "otisfin",
        "ckan_resource_id": "",
    },
    # ── NAPS (Environment Canada National Air Pollution Surveillance) ─────
    # Fetched on demand via moirais.earth.fetch_naps (no auth; Open-Canada CKAN).
    # Data lands at ~/.cache/moirais/earth/ as parquet (keyed by query hash).
    # local_path kept for compatibility but not used; loader dispatches via
    # the 'fetcher' field when entry['source'] == 'naps'.
    "naps-no2-on-2023": {
        "name": "NAPS NO2 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_on_2023",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "ON"},
    },
    "naps-pm25-on-2023": {
        "name": "NAPS PM2.5 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm25_on_2023",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "ON"},
    },
    "naps-o3-on-2023": {
        "name": "NAPS O3 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_o3_on_2023",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2023, "province": "ON"},
    },
    "naps-so2-on-2023": {
        "name": "NAPS SO2 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_so2_on_2023",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "so2", "year": 2023, "province": "ON"},
    },
    "naps-co-on-2023": {
        "name": "NAPS CO Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_co_on_2023",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "co", "year": 2023, "province": "ON"},
    },
    "naps-pm10-on-2023": {
        "name": "NAPS PM10 Ontario 2023",
        "source": "naps",
        "survey": "naps",
        "year": "2023",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_pm10_on_2023",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm10", "year": 2023, "province": "ON"},
    },
    "naps-no2-on-2022": {
        "name": "NAPS NO2 Ontario 2022 (trend baseline)",
        "source": "naps",
        "survey": "naps",
        "year": "2022",
        "format": "fetcher",
        "type": "air-quality",
        "large_file": False,
        "local_path": "",
        "table_name": "naps_no2_on_2022",
        "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2022, "province": "ON"},
    },

    # ────────────────────────────────────────────────────────────────
    # W5 extension (2026-04-17 night): other provinces + trend years.
    # Each entry dispatches to moirais.earth.fetch_naps via the 'fetcher'
    # field, so adding another province/year is one dict entry and
    # NAPS CKAN auto-caches under ~/.cache/moirais/earth/ as parquet.
    # ────────────────────────────────────────────────────────────────

    # --- 2023, other key provinces (PM2.5 + NO2) ---
    "naps-pm25-qc-2023": {
        "name": "NAPS PM2.5 Quebec 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_qc_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "QC"},
    },
    "naps-no2-qc-2023": {
        "name": "NAPS NO2 Quebec 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_no2_qc_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "QC"},
    },
    "naps-pm25-bc-2023": {
        "name": "NAPS PM2.5 British Columbia 2023 (wildfire-smoke dominant)",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_bc_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "BC"},
    },
    "naps-no2-bc-2023": {
        "name": "NAPS NO2 British Columbia 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_no2_bc_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "BC"},
    },
    "naps-pm25-ab-2023": {
        "name": "NAPS PM2.5 Alberta 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_ab_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "AB"},
    },
    "naps-no2-ab-2023": {
        "name": "NAPS NO2 Alberta 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_no2_ab_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "AB"},
    },
    "naps-pm25-ns-2023": {
        "name": "NAPS PM2.5 Nova Scotia 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_ns_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": "NS"},
    },
    "naps-no2-ns-2023": {
        "name": "NAPS NO2 Nova Scotia 2023",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_no2_ns_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": "NS"},
    },

    # --- Ontario PM2.5 trend baseline (2019-2022) ---
    "naps-pm25-on-2022": {
        "name": "NAPS PM2.5 Ontario 2022 (trend baseline)",
        "source": "naps", "survey": "naps", "year": "2022", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_on_2022", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2022, "province": "ON"},
    },
    "naps-pm25-on-2021": {
        "name": "NAPS PM2.5 Ontario 2021 (pandemic-year baseline)",
        "source": "naps", "survey": "naps", "year": "2021", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_on_2021", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2021, "province": "ON"},
    },
    "naps-pm25-on-2020": {
        "name": "NAPS PM2.5 Ontario 2020 (COVID-lockdown air quality)",
        "source": "naps", "survey": "naps", "year": "2020", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_on_2020", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2020, "province": "ON"},
    },
    "naps-pm25-on-2019": {
        "name": "NAPS PM2.5 Ontario 2019 (pre-COVID reference)",
        "source": "naps", "survey": "naps", "year": "2019", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_on_2019", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2019, "province": "ON"},
    },

    # --- Ontario O3 trend (for heat × ozone interaction studies) ---
    "naps-o3-on-2022": {
        "name": "NAPS O3 Ontario 2022 (heat × ozone study base)",
        "source": "naps", "survey": "naps", "year": "2022", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_o3_on_2022", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2022, "province": "ON"},
    },
    "naps-o3-on-2021": {
        "name": "NAPS O3 Ontario 2021",
        "source": "naps", "survey": "naps", "year": "2021", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_o3_on_2021", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2021, "province": "ON"},
    },

    # --- 2023 national (no province filter; all-Canada aggregate) ---
    "naps-pm25-ca-2023": {
        "name": "NAPS PM2.5 Canada 2023 (national, all provinces)",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_pm25_ca_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "pm25", "year": 2023, "province": None},
    },
    "naps-no2-ca-2023": {
        "name": "NAPS NO2 Canada 2023 (national, all provinces)",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_no2_ca_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "no2", "year": 2023, "province": None},
    },
    "naps-o3-ca-2023": {
        "name": "NAPS O3 Canada 2023 (national, all provinces)",
        "source": "naps", "survey": "naps", "year": "2023", "format": "fetcher",
        "type": "air-quality", "large_file": False, "local_path": "",
        "table_name": "naps_o3_ca_2023", "ckan_resource_id": "",
        "fetcher": "moirais.earth:fetch_naps",
        "fetcher_args": {"pollutant": "o3", "year": 2023, "province": None},
    },
}


_METADATA_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS _moirais_metadata (
    table_name TEXT PRIMARY KEY,
    source TEXT,
    survey TEXT,
    year TEXT,
    format TEXT,
    row_count INTEGER,
    col_count INTEGER,
    columns TEXT,
    ingested_at TEXT,
    file_hash TEXT
)
"""


# ---------------------------------------------------------------------------
# Built-in database — ships with the package
# ---------------------------------------------------------------------------


def moirais_db() -> Path:
    """Return path to moirais.db — checks package-bundled location first, then cache.

    Package-bundled DB ships with the installed package (via LFS/setuptools).
    Cache location is used for development and is gitignored.
    """
    # 1. Package-bundled DB (ships with moirais package, available in CI/install)
    package_db = Path(__file__).parent / "data" / "moirais.db"
    if package_db.exists():
        return package_db
    # 2. Project cache location (development only, gitignored)
    return _project_root() / "data" / "cache" / "moirais.db"


def _builtin_db_connect() -> sqlite3.Connection | None:
    """Connect to the built-in database if it exists."""
    db_path = moirais_db()
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    return None


# ---------------------------------------------------------------------------
# SQLite cache — shared between Python and R via DBI
# ---------------------------------------------------------------------------


def _resolve_cache_path(db_path: str | Path | None = None) -> Path:
    """Resolve the cache database path, creating parent dirs as needed."""
    p = Path(db_path or os.environ.get("MOIRAIS_CACHE_DB", DEFAULT_CACHE_DB))
    if not p.is_absolute():
        p = _project_root() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def cache_connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (or create) the MOIRAIS SQLite cache database."""
    p = _resolve_cache_path(db_path)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(_METADATA_TABLE_DDL)
    return conn


def cache_store(df: pd.DataFrame, table: str, db_path: str | Path | None = None) -> int:
    """Write a DataFrame to the SQLite cache, replacing any existing table."""
    conn = cache_connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
        n = len(df)
        logger.info("Cached %d rows → %s", n, table)
        return n
    finally:
        conn.close()


def cache_load(table: str, db_path: str | Path | None = None) -> pd.DataFrame | None:
    """Load a table from the SQLite cache. Returns None if not cached."""
    conn = cache_connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not tables:
            return None
        return pd.read_sql(f"SELECT * FROM [{_safe_table_name(table)}]", conn)
    finally:
        conn.close()


def cache_list(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """List all cached tables with row counts."""
    conn = cache_connect(db_path)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        result = []
        for (name,) in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM [{_safe_table_name(name)}]").fetchone()[0]
            result.append({"table": name, "rows": count})
        return result
    finally:
        conn.close()


def fetch_ckan_to_cache(
    dataset_key: str = "cpads",
    limit: int = 32000,
    db_path: str | Path | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch a dataset from CKAN and store it in the SQLite cache.

    Parameters
    ----------
    dataset_key : str
        Key in CKAN_DATASETS (e.g., "cpads", "csads", "csus").
    limit : int
        Max records to fetch from CKAN DataStore API.
    db_path : str | Path | None
        Override cache database path.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    pd.DataFrame
        The fetched (and optionally canonicalized) DataFrame.
    """
    info = CKAN_DATASETS.get(dataset_key)
    if not info:
        raise ValueError(f"Unknown CKAN dataset: {dataset_key}. Known: {list(CKAN_DATASETS)}")

    resource_id = info["resource_id"]
    if not resource_id:
        # Resolve resource_id from package metadata.
        meta_url = info["metadata_url"]
        logger.info("Resolving resource_id from %s", meta_url)
        meta = json.loads(urlopen(meta_url, timeout=timeout).read().decode())
        resources = meta.get("result", {}).get("resources", [])
        csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV"]
        if csv_resources:
            resource_id = csv_resources[0]["id"]
        elif resources:
            resource_id = resources[0]["id"]
        else:
            raise ValueError(f"No resources found for {dataset_key}")

    params = {"resource_id": resource_id, "limit": limit}
    url = f"{DEFAULT_CKAN_API_BASE}?{urlencode(params)}"
    logger.info("Fetching %s from CKAN (%d records max)...", dataset_key, limit)
    raw = urlopen(url, timeout=timeout).read().decode()
    payload = json.loads(raw)
    records = payload.get("result", {}).get("records", [])
    if not records:
        raise RuntimeError(f"CKAN returned 0 records for {dataset_key}")

    df = pd.DataFrame.from_records(records)
    # Drop CKAN internal column.
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    logger.info("Fetched %d rows x %d cols for %s", len(df), len(df.columns), dataset_key)

    # Cache raw data.
    table_name = f"{dataset_key}_raw"
    cache_store(df, table_name, db_path)

    # If CPADS, also canonicalize and cache the canonical version.
    if dataset_key == "cpads" and has_raw_cpads_columns(df):
        canonical = canonicalize_cpads_frame(df)
        cache_store(canonical, "cpads_canonical", db_path)
        return canonical

    return df


def load_cpads(db_path: str | Path | None = None, timeout: int = 60) -> pd.DataFrame:
    """Load CPADS data: try local files, then cache, then CKAN API.

    Resolution order:
    1. Local RDS (via R bridge) or CSV files in standard locations
    2. SQLite cache (data/cache/moirais.db)
    3. CKAN API fetch → cache → return
    """
    # 1. Try local files.
    local_candidates = [
        "data/datasets/oc/CPADS/2021-2022/cpads-2021-2022-pumf2.csv",
        "data/cache/cpads_pumf_wrangled.rds",
    ]
    for path in local_candidates:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix == ".csv":
            df = pd.read_csv(p)
            if has_raw_cpads_columns(df):
                return canonicalize_cpads_frame(df)
            validate_cpads_frame(df, strict=True)
            return df
        if p.suffix == ".rds":
            # RDS files need R — skip in Python, prefer CSV or cache.
            continue

    # 2. Try built-in moirais.db (ships with package).
    builtin = _builtin_db_connect()
    if builtin is not None:
        try:
            # Look for the canonical CPADS table in the built-in DB
            for tbl in ("cpads_canonical", "ocp21"):
                hit = builtin.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (tbl,),
                ).fetchone()
                if hit:
                    df = pd.read_sql(f"SELECT * FROM [{_safe_table_name(tbl)}]", builtin)
                    logger.info("Loaded CPADS from built-in DB table %s (%d rows)", tbl, len(df))
                    if has_raw_cpads_columns(df):
                        return canonicalize_cpads_frame(df)
                    return df
        finally:
            builtin.close()

    # 3. Try SQLite cache.
    cached = cache_load("cpads_canonical", db_path)
    if cached is not None:
        logger.info("Loaded CPADS from cache (%d rows)", len(cached))
        return cached

    # 4. Fetch from CKAN API and cache.
    logger.info("CPADS not found locally or in cache. Fetching from CKAN...")
    return fetch_ckan_to_cache("cpads", db_path=db_path, timeout=timeout)


# ---------------------------------------------------------------------------
# Unified load interface
# ---------------------------------------------------------------------------


def _fuzzy_match_key(key: str) -> str | None:
    """Match a key like 'cpads', 'ocp21', or 'naps-no2-on-2023' to a
    catalog entry. Handles both hyphen and underscore separators in the
    input by normalising BOTH the query and each catalog key to the same
    form (underscores + lowercase + no whitespace) before comparing."""
    def _norm(s: str) -> str:
        return s.lower().replace("-", "_").replace(" ", "")

    key_lower = _norm(key)
    # Exact match on either the raw key or the normalised form.
    if key in DATASET_CATALOG:
        return key
    if key_lower in DATASET_CATALOG:
        return key_lower
    # Normalised equality against every catalog key.
    for full_key in DATASET_CATALOG:
        if _norm(full_key) == key_lower:
            return full_key
    # Substring matches on key / survey / name (also normalised).
    for full_key, entry in DATASET_CATALOG.items():
        if key_lower in _norm(full_key):
            return full_key
        if key_lower == _norm(entry.get("survey", "")):
            return full_key
        if key_lower in entry.get("name", "").lower().replace(" ", ""):
            return full_key
    return None


def load_dataset(
    key: str,
    *,
    db_path: str | Path | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Load a dataset by catalog key.

    Resolution order:
    1. Built-in moirais.db (ships with package)
    2. User cache data/cache/moirais.db
    3. Local file (ingest to cache on the fly)
    4. CKAN API (if resource ID available)
    5. Error

    Supports fuzzy matching: ``load_dataset("cpads")`` resolves to ``ocp21``.
    """
    matched = _fuzzy_match_key(key)
    if matched is None:
        available = ", ".join(sorted(DATASET_CATALOG))
        raise KeyError(f"Unknown dataset key: {key!r}. Available: {available}")

    entry = DATASET_CATALOG[matched]
    table_name = entry["table_name"]

    # 1. Built-in database (ships with package).
    builtin = _builtin_db_connect()
    if builtin is not None:
        try:
            tables = builtin.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            ).fetchone()
            if tables:
                df = pd.read_sql(f"SELECT * FROM [{_safe_table_name(table_name)}]", builtin)
                logger.info("Loaded %s from built-in DB (%d rows)", matched, len(df))
                return df
        finally:
            builtin.close()

    # 2. User cache.
    cached = cache_load(table_name, db_path)
    if cached is not None:
        logger.info("Loaded %s from cache (%d rows)", matched, len(cached))
        return cached

    # 2b. Fetcher dispatch (NAPS, OpenAQ, Earth Engine, ArcGIS).
    #     Entry has ``source`` in {"naps", "openaq", "ee", "arcgis"} OR a
    #     ``fetcher`` key of the form "module.path:fn_name". Call the fn
    #     with ``fetcher_args`` and cache the resulting DataFrame.
    fetcher_spec = entry.get("fetcher")
    if fetcher_spec:
        import importlib
        mod_name, fn_name = fetcher_spec.split(":", 1)
        try:
            mod = importlib.import_module(mod_name)
            fetcher_fn = getattr(mod, fn_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                f"Cannot resolve fetcher {fetcher_spec!r} for dataset "
                f"{matched!r}: {exc}"
            ) from exc
        args = dict(entry.get("fetcher_args") or {})
        logger.info("Fetching %s via %s(**%s) ...", matched, fetcher_spec, args)
        df = fetcher_fn(**args)
        if df is not None and len(df):
            cache_store(df, table_name, db_path)
        return df

    # 3. Local file.
    local_path = Path(entry["local_path"])
    if not local_path.is_absolute():
        local_path = _project_root() / local_path
    if local_path.exists():
        logger.info("Ingesting %s from local file: %s", matched, local_path)
        if entry["format"] == "csv":
            df = pd.read_csv(local_path, low_memory=False)
        elif entry["format"] == "xlsx":
            df = pd.read_excel(local_path)
        else:
            raise NotImplementedError(f"Format {entry['format']} not supported for on-the-fly ingest")
        cache_store(df, table_name, db_path)
        return df

    # 4. CKAN API.
    rid = entry.get("ckan_resource_id", "")
    if rid:
        logger.info("Fetching %s from CKAN API...", matched)
        return fetch_ckan_to_cache(matched, db_path=db_path, timeout=timeout)

    raise FileNotFoundError(
        f"Dataset {matched!r} not found in cache, at {entry['local_path']}, "
        f"or via CKAN. Run: python libexec/config/tests/rtests/ingest_datasets.py --only {matched}"
    )


def list_datasets(db_path: str | Path | None = None) -> list[dict]:
    """List all datasets with their cache status.

    Returns a list of dicts with keys: key, name, source, survey, year,
    type, cached (bool), rows (int or None).

    When *db_path* is explicitly provided, only that database is queried
    (the built-in DB is skipped).  This keeps unit tests deterministic.
    """
    cached_tables: dict[str, int] = {}
    if db_path is None:
        # Check built-in database first (only when no explicit db_path)
        builtin = _builtin_db_connect()
        if builtin is not None:
            try:
                for (name,) in builtin.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                    count = builtin.execute(f"SELECT COUNT(*) FROM [{_safe_table_name(name)}]").fetchone()[0]
                    cached_tables[name] = count
            except Exception:
                pass
            finally:
                builtin.close()
    # Also check user cache (or the explicit db_path)
    try:
        for item in cache_list(db_path):
            if item["table"] not in cached_tables:
                cached_tables[item["table"]] = item["rows"]
    except Exception:
        pass

    result = []
    for key, entry in DATASET_CATALOG.items():
        tbl = entry["table_name"]
        result.append(
            {
                "key": key,
                "name": entry["name"],
                "source": entry["source"],
                "survey": entry["survey"],
                "year": entry["year"],
                "type": entry["type"],
                "cached": tbl in cached_tables,
                "rows": cached_tables.get(tbl),
            }
        )
    return result


def dataset_info(key: str) -> dict:
    """Return full metadata for a dataset by catalog key."""
    matched = _fuzzy_match_key(key)
    if matched is None:
        raise KeyError(f"Unknown dataset key: {key!r}")
    info = dict(DATASET_CATALOG[matched])
    info["key"] = matched
    info["local_exists"] = Path(info["local_path"]).exists()
    cached = cache_load(info["table_name"])
    info["cached"] = cached is not None
    info["cached_rows"] = len(cached) if cached is not None else None
    return info


class DatasetRegistry:
    """
    A registry to manage, catalog, and load secure epidemiological datasets.
    """

    def __init__(self, data_dir: str = "data/datasets/"):
        """
        Initialize the dataset registry.

        :param data_dir: The root file directory containing secure datasets, defaults to "data/datasets/".
        :type data_dir: str
        """
        self.data_dir = os.fspath(data_dir)
        self.catalog = {
            "cpads_2021_2022_pumf": {
                "name": "CPADS 2021-2022 PUMF",
                "path": "cpads/2021_2022/pumf.csv",
                "format": "csv",
                "type": "survey",
                "source_kind": "local_private_file",
                "landing_page": "https://open.canada.ca/data/en/dataset/736fa9b2-62e4-4e31-aea4-51869605b363",
                "documentation_url": "https://open.canada.ca/data/en/dataset/736fa9b2-62e4-4e31-aea4-51869605b363",
                "ckan_api_base": DEFAULT_CKAN_API_BASE,
                "ckan_resource_id": "d2639429-c304-45a6-90b3-770562f4d46d",
                "required_variables": list(CPADS_REQUIRED_VARIABLES),
            }
        }

    def register_dataset(self, name: str, metadata: dict[str, Any]):
        """
        Register a dataset in the internal catalog.

        :param name: Unique identifier for the dataset.
        :type name: str
        :param metadata: A dictionary mapping metadata values (like name, path, format).
        :type metadata: dict
        """
        required_keys = {"name", "path", "format", "type"}
        missing_keys = required_keys.difference(metadata)
        if missing_keys:
            raise ValueError("Dataset metadata is missing required keys: " + ", ".join(sorted(missing_keys)))

        self.catalog[name] = dict(metadata)

    def list_datasets(self) -> dict[str, dict[str, Any]]:
        """
        Return a copy of the registered dataset catalog.

        :return: Dataset metadata keyed by registry identifier.
        :rtype: dict[str, dict[str, Any]]
        """
        return deepcopy(self.catalog)

    def get_dataset_metadata(self, name: str) -> dict[str, Any]:
        """Return a copy of the metadata for one registered dataset."""
        if name not in self.catalog:
            raise ValueError(f"Dataset {name} not found in registry.")
        return dict(self.catalog[name])

    def fetch_ckan_records(
        self,
        name: str,
        *,
        limit: int = 5,
        query: str | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Fetch records for a dataset backed by a CKAN DataStore resource.
        """
        info = self.get_dataset_metadata(name)
        resource_id = info.get("ckan_resource_id")
        if not resource_id:
            raise ValueError(f"Dataset {name} does not define a CKAN resource id.")

        params = {
            "resource_id": resource_id,
            "limit": int(limit),
        }
        if query:
            params["q"] = query

        base_url = info.get("ckan_api_base", DEFAULT_CKAN_API_BASE)
        url = f"{base_url}?{urlencode(params)}"
        payload = urlopen(url, timeout=timeout).read().decode("utf-8")
        return json.loads(payload)

    def fetch_ckan_dataframe(
        self,
        name: str,
        *,
        limit: int = 100,
        query: str | None = None,
        timeout: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch a CKAN result set and return the records as a DataFrame.
        """
        payload = self.fetch_ckan_records(
            name,
            limit=limit,
            query=query,
            timeout=timeout,
        )
        records = payload.get("result", {}).get("records", [])
        return pd.DataFrame.from_records(records)

    def cpads_contract(self) -> dict[str, Any]:
        """Return the canonical local-private CPADS contract."""
        return cpads_contract()

    def register_local_cpads(self, path: str | Path, *, name: str = "cpads_local") -> dict[str, Any]:
        """Register a user-provided local CPADS file."""
        resolved = Path(path).expanduser()
        metadata = {
            "name": "Local CPADS analysis file",
            "path": os.fspath(resolved),
            "format": infer_file_format(resolved),
            "type": "survey",
            "source_kind": "local_private_file",
            "required_variables": list(CPADS_REQUIRED_VARIABLES),
        }
        self.catalog[name] = metadata
        return dict(metadata)

    def validate_cpads_frame(self, frame: pd.DataFrame, *, strict: bool = True) -> list[str]:
        """Validate a DataFrame against the canonical CPADS variable contract."""
        return validate_cpads_frame(frame, strict=strict)

    def load(self, name: str) -> pd.DataFrame:
        """
        Load a registered dataset securely.

        :param name: Unique identifier for the dataset in the catalog.
        :type name: str
        :raises ValueError: If the dataset name is not in the registry catalog.
        :raises FileNotFoundError: If the underlying file could not be queried or synced locally.
        :raises NotImplementedError: If the specified file format is not supported.
        :return: A pandas DataFrame containing the dataset content.
        :rtype: pandas.DataFrame
        """
        if name not in self.catalog:
            raise ValueError(f"Dataset {name} not found in registry.")

        info = self.catalog[name]
        raw_path = info["path"]
        path = raw_path if os.path.isabs(raw_path) else os.path.join(self.data_dir, raw_path)

        # In a real environment, we would securely fetch via requests
        # but here we mock the filesystem load for the beta release
        if not os.path.exists(path):
            raise FileNotFoundError(f"Underlying file {path} not synced to {self.data_dir}")

        if info["format"] == "csv":
            frame = pd.read_csv(path)
        elif info["format"] == "excel":
            frame = pd.read_excel(path, engine="openpyxl")
        elif info["format"] == "rds":
            raise NotImplementedError(
                "RDS loading is not supported from Python. Provide a CSV/Excel export or use the R package."
            )
        else:
            raise NotImplementedError("Format not supported.")

        required_variables = info.get("required_variables")
        if required_variables:
            if has_raw_cpads_columns(frame):
                frame = canonicalize_cpads_frame(frame)
            else:
                validate_cpads_frame(frame, strict=True)
        return frame
