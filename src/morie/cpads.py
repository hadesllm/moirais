"""CPADS data contract helpers for local-private analysis."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

CPADS_REQUIRED_VARIABLES = [
    "weight",
    "alcohol_past12m",
    "heavy_drinking_30d",
    "ebac_tot",
    "ebac_legal",
    "cannabis_any_use",
    "age_group",
    "gender",
    "province_region",
    "mental_health",
    "physical_health",
]

CPADS_RAW_COLUMN_MAP = {
    "weight": "wtpumf",
    "alcohol_past12m": "alc05",
    "heavy_drinking_30d_total": "alc12_30d_prev_total",
    "heavy_drinking_30d_fallback": "alc12_30d_prev",
    "cannabis_any_use": "can05",
    "age_group": "age_groups",
    "gender": "dvdemq01",
    "province_region": "region",
    "mental_health": "hwbq02",
    "physical_health": "hwbq01",
    "ebac_tot": "ebac_tot",
    "ebac_legal": "ebac_legal",
}


def cpads_contract() -> dict[str, object]:
    """Return the local-private CPADS contract used by MORIE workflows."""
    return {
        "source_kind": "local_private_file",
        "expected_wrangled_path": "data/cache/cpads_pumf_wrangled.rds",
        "required_variables": list(CPADS_REQUIRED_VARIABLES),
        "raw_column_map": dict(CPADS_RAW_COLUMN_MAP),
        "note": "CPADS row-level data must be supplied locally and must not be committed to git.",
    }


def missing_cpads_variables(columns: Iterable[str]) -> list[str]:
    """Return missing canonical CPADS variables."""
    columns = set(columns)
    return [name for name in CPADS_REQUIRED_VARIABLES if name not in columns]


def validate_cpads_frame(frame: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """Validate that a frame contains the canonical CPADS analysis columns."""
    missing = missing_cpads_variables(frame.columns)
    if strict and missing:
        raise ValueError("CPADS frame is missing required variables: " + ", ".join(missing))
    return missing


def has_raw_cpads_columns(frame: pd.DataFrame) -> bool:
    """Return True when the frame looks like raw CPADS PUMF data.

    Accepts both ``wtpumf`` (PUMF release) and ``wtdf`` (full dataset)
    as the weight column.
    """
    raw_cols = set(CPADS_RAW_COLUMN_MAP.values())
    # Allow either weight column variant.
    raw_cols_alt = (raw_cols - {"wtpumf"}) | {"wtdf"}
    return raw_cols.issubset(frame.columns) or raw_cols_alt.issubset(frame.columns)


def canonicalize_cpads_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw CPADS PUMF columns into the canonical MORIE analysis columns.

    This is a first-pass canonicalization layer based on the current public
    CPADS PUMF field names present in the repository CSV.
    """
    if not has_raw_cpads_columns(frame):
        validate_cpads_frame(frame, strict=True)
        return frame.copy()

    out = frame.copy()
    # Accept both weight column names: wtpumf (PUMF release) and wtdf (full dataset).
    weight_col = "wtpumf" if "wtpumf" in frame.columns else "wtdf"
    out["weight"] = pd.to_numeric(frame[weight_col], errors="coerce")
    out["alcohol_past12m"] = frame["alc05"].replace({1: 1, 2: 0, 98: np.nan, 99: np.nan})
    out["heavy_drinking_30d"] = np.where(
        frame["alc12_30d_prev_total"] == 1,
        1,
        np.where(
            frame["alc12_30d_prev_total"] == 0,
            0,
            np.where(
                frame["alc12_30d_prev"] == 1,
                1,
                np.where(frame["alc12_30d_prev"] == 0, 0, np.nan),
            ),
        ),
    )
    out["cannabis_any_use"] = frame["can05"].replace({1: 1, 2: 0, 98: np.nan, 99: np.nan})
    out["age_group"] = frame["age_groups"].replace({98: np.nan, 99: np.nan})
    out["gender"] = frame["dvdemq01"].replace({98: np.nan, 99: np.nan})
    out["province_region"] = frame["region"].replace({98: np.nan, 99: np.nan})
    out["mental_health"] = frame["hwbq02"].replace({98: np.nan, 99: np.nan})
    out["physical_health"] = frame["hwbq01"].replace({98: np.nan, 99: np.nan})
    out["ebac_tot"] = pd.to_numeric(frame["ebac_tot"], errors="coerce")
    out["ebac_legal"] = pd.to_numeric(frame["ebac_legal"], errors="coerce")
    validate_cpads_frame(out, strict=True)
    return out


def infer_file_format(path: str | Path) -> str:
    """Infer a dataset format from a local path."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    if suffix == ".rds":
        return "rds"
    raise ValueError(f"Unsupported CPADS file format for path: {path}")
