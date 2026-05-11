"""OTIS analysis surfaces — RichResult wrappers around morie.otis primitives.

`morie.otis` provides 6 result-emitting callables that return specialized
dataclasses (RplRes / AstRes / VolRes / OtDmlR / etc.).  This module wraps
each in a `RichResult` so OTIS analyses get the same multi-section output
that Welch / Mann–Whitney / SIU analyses already emit.

OTIS = anonymized Ontario MCSCS correctional placement records (open
data). The canonical 76,934-row table lives in the repo cache at
`data/cache/correctional_stats_report_environment.RData` as the R object
`df`. Use `load_otis()` (CSV mirror) or call the Rscript exporter at
`scripts/export_otis_csv.R` to refresh the CSV.

Public API:
    rplace(df, year, sex=None) -> RichResult
    astcmb(df) -> RichResult
    volat(df) -> RichResult
    rctrnd(df) -> RichResult
    otdesc(df, year=None, sex=None) -> RichResult
    otdml(df, treatment, outcome, ...) -> RichResult
    all_analyses(df, year, *, out_dir=None) -> dict[str, RichResult]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import otis as _o
from .fn._richresult import RichResult


PROJECT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = PROJECT / "data/manifest/outputs/otis"

# OTIS is open, anonymized Ontario MCSCS correctional-placement data.
# The R workflow (libexec/config/tests/rtests/otis.R) loads
# data/cache/correctional_stats_report_environment.RData and exposes
# `df` (76,934 rows × 10 cols) as the canonical OTIS table. Our loader
# reads the Rscript-exported CSV mirror at data/cache/otis_main.csv.
DEFAULT_OTIS_CSV = PROJECT / "data/cache/otis_main.csv"
DEFAULT_OTIS_DT_CSV = PROJECT / "data/cache/otis_dt.csv"


def load_otis(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the canonical OTIS DataFrame.

    Defaults to data/cache/otis_main.csv (Rscript-exported from the
    correctional_stats_report_environment.RData fixture). Pass a custom
    `csv_path` to load a refreshed snapshot.

    Returns a 10-column DataFrame:
        end_fiscal_year, unique_individual_id, region_at_time_of_placement,
        region_most_recent_placement, gender, age_category,
        mental_health_alert, suicide_risk_alert, suicide_watch_alert,
        number_of_placements
    """
    p = Path(csv_path) if csv_path else DEFAULT_OTIS_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"OTIS dataset not found at {p}. Re-run "
            "Rscript scripts/export_otis_csv.R or restore the .RData "
            "fixture under data/cache/."
        )
    return pd.read_csv(p)


def rplace(df: pd.DataFrame, year: int, sex: str | None = None,
           **kw: Any) -> RichResult:
    """Regional placement analysis as RichResult."""
    r = _o.rplace(df, year=year, sex=sex, **kw)
    return RichResult(
        title=f"OTIS regional placement — fiscal year {year}"
              + (f" (sex={sex})" if sex else ""),
        summary_lines=[
            ("Year", year),
            ("Sex filter", sex or "all"),
            ("Age groups", ", ".join(map(str, r.counts.index.tolist()))),
            ("Regions", ", ".join(map(str, r.counts.columns.tolist()))),
            ("Total individuals (cells summed)", int(r.counts.values.sum())),
        ],
        tables=[
            {"title": "Counts (age × region):",
             "headers": ["age"] + list(map(str, r.counts.columns)),
             "rows": [[idx] + list(map(int, row))
                      for idx, row in r.counts.iterrows()]},
            {"title": "Proportions (within age):",
             "headers": ["age"] + list(map(str, r.props.columns)),
             "rows": [[idx] + [f"{v*100:.1f}%" for v in row]
                      for idx, row in r.props.iterrows()]},
        ],
        payload={"counts": r.counts.to_dict(),
                 "props": r.props.to_dict(),
                 "year": year, "sex": sex},
    )


def astcmb(df: pd.DataFrame, **kw: Any) -> RichResult:
    """Alert-state combination analysis as RichResult."""
    r = _o.astcmb(df, **kw)
    summary = r.summary
    return RichResult(
        title="OTIS alert-state combination encoding",
        summary_lines=[
            ("Individuals classified", int(r.data.shape[0])),
            ("Combination categories", int(summary.shape[0])),
        ],
        tables=[{
            "title": "Count by alert-complexity level:",
            "headers": ["combo"] + list(map(str, summary.columns)),
            "rows": [[idx] + list(row.tolist())
                     for idx, row in summary.iterrows()],
        }],
        interpretation=(
            "Each individual is encoded as one of 8 combinations of "
            "(mental_health, suicide_risk, suicide_watch). a8=no alerts, "
            "a7=all three. Higher complexity = more concurrent risk flags."
        ),
        payload={"summary": summary.to_dict()},
    )


def volat(df: pd.DataFrame, **kw: Any) -> RichResult:
    """Regional volatility metric as RichResult."""
    r = _o.volat(df, **kw)
    return RichResult(
        title="OTIS regional volatility (placement movement)",
        summary_lines=[
            ("Individuals scored", int(r.data.shape[0])),
            ("Mean volatility", float(r.mean)),
            ("Median volatility", float(r.median)),
            ("Min", float(r.data.iloc[:, -1].min())
                if r.data.shape[0] else float("nan")),
            ("Max", float(r.data.iloc[:, -1].max())
                if r.data.shape[0] else float("nan")),
        ],
        interpretation=(
            "Volatility = number of distinct regions a person passed "
            "through across the OTIS observation window, normalised to "
            "[0,1]. Higher = more movement between regional facilities."
        ),
        payload={"mean": float(r.mean), "median": float(r.median),
                 "n": int(r.data.shape[0])},
    )


def rctrnd(df: pd.DataFrame, **kw: Any) -> RichResult:
    """Restrictive-confinement trends as RichResult."""
    r = _o.rctrnd(df, **kw)
    # _o.rctrnd returns a DataFrame
    return RichResult(
        title="OTIS restrictive-confinement trends over time",
        summary_lines=[
            ("Years observed", f"{int(r['end_fiscal_year'].min())}–"
             f"{int(r['end_fiscal_year'].max())}"
             if "end_fiscal_year" in r.columns and r.shape[0]
             else "n/a"),
            ("Rows", int(r.shape[0])),
        ],
        tables=[{
            "title": "Restrictive-confinement counts by year:",
            "headers": list(map(str, r.columns)),
            "rows": [list(row.tolist()) for _, row in r.iterrows()],
        }],
        payload={"trends": r.to_dict("records")},
    )


def otdesc(df: pd.DataFrame, **kw: Any) -> RichResult:
    """Full OTIS descriptive suite as RichResult.

    `_o.otdesc` does not accept year/sex filters — apply those upstream
    via `df = df[df.end_fiscal_year == year]` if needed.
    """
    r = _o.otdesc(df, **kw)
    # otdesc returns a dict-of-DataFrames. Wrap each as a sub-table.
    tables = []
    for key, val in (r.items() if isinstance(r, dict) else []):
        if isinstance(val, pd.DataFrame):
            tables.append({
                "title": f"{key}:",
                "headers": list(map(str, val.columns)),
                "rows": [list(row.tolist()) for _, row in val.iterrows()],
            })
    return RichResult(
        title="OTIS descriptives",
        summary_lines=[
            ("Total individuals (unique)", int(r.get("n_total", 0))
             if isinstance(r, dict) else 0),
            ("Total records", int(r.get("n_records", 0))
             if isinstance(r, dict) else 0),
            ("Tables emitted", len(tables)),
        ],
        tables=tables,
    )


def otdml(df: pd.DataFrame, *, treatment: str, outcome: str,
          covariates: list[str], **kw: Any) -> RichResult:
    """OTIS DML IRM (ATE/ATT) as RichResult."""
    r = _o.otdml(df, treatment=treatment, outcome=outcome,
                 covariates=covariates, **kw)
    return RichResult(
        title=f"OTIS DML {r.method}: ATE/ATT for {treatment} → {outcome}",
        summary_lines=[
            ("Method", r.method),
            ("Treatment", treatment),
            ("Outcome", outcome),
            ("Covariates", ", ".join(covariates)),
            ("n", int(r.n)),
        ],
        tables=[{
            "title": "Causal effect estimates:",
            "headers": ["Estimand", "Estimate", "SE", "p-value"],
            "rows": [
                ["ATE", f"{r.ate:.4f}", f"{r.ate_se:.4f}",
                 f"{r.ate_pval:.4g}"],
                ["ATT", f"{r.att:.4f}", f"{r.att_se:.4f}",
                 f"{r.att_pval:.4g}"],
            ],
        }],
        interpretation=(
            f"ATE = average effect of {treatment} on {outcome} across "
            "the entire OTIS population. ATT = average effect among "
            "those who actually received the treatment. p-values use "
            "the DML asymptotic-normal approximation."
        ),
        payload={"ate": r.ate, "ate_se": r.ate_se, "ate_pval": r.ate_pval,
                 "att": r.att, "att_se": r.att_se, "att_pval": r.att_pval,
                 "n": int(r.n), "method": r.method},
    )


def all_analyses(df: pd.DataFrame, year: int,
                 *, sex: str | None = None,
                 out_dir: Path | None = None) -> dict[str, RichResult]:
    """Run rplace / astcmb / volat / rctrnd / otdesc on `df` and write to disk.

    `otdml` is excluded from the bundle because it requires the user
    to specify (treatment, outcome, covariates) — call it separately.
    """
    out_dir = out_dir or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    for name, fn in [
        ("rplace", lambda: rplace(df, year=year, sex=sex)),
        ("astcmb", lambda: astcmb(df)),
        ("volat", lambda: volat(df)),
        ("rctrnd", lambda: rctrnd(df)),
        ("otdesc", lambda: otdesc(df)),
    ]:
        try:
            r = fn()
            results[name] = r
            (out_dir / f"otis_analysis_{name}.txt").write_text(str(r))
            (out_dir / f"otis_analysis_{name}.json").write_text(
                json.dumps(r.payload, indent=2, default=str, ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            results[name] = RichResult(
                title=f"otis.{name} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
