"""SIU analysis surfaces — turns the scraped SIU.csv / SIU_by_case.csv
into structured, RichResult-emitting analyses.

Each callable here loads the canonical SIU outputs from
data/datasets/vsr/SIU_by_case.csv (or accepts a DataFrame directly)
and emits a `RichResult` with table + warnings + interpretation.

Design rule: every analysis should be reproducible from the canonical
inputs and produce CSV/JSON output under data/manifest/outputs/siu/.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from ..fn._richresult import RichResult


PROJECT = Path(__file__).resolve().parents[6]
DEFAULT_CSV = PROJECT / "data/datasets/vsr/SIU_by_case.csv"
DEFAULT_OUT = PROJECT / "data/manifest/outputs/siu"


def _load(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load SIU_by_case.csv (or whichever path is given)."""
    p = Path(csv_path) if csv_path else DEFAULT_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"SIU dataset not found at {p}. Run scripts/scrape_siu_full.py "
            "and scripts/reparse_siu_cache.py first."
        )
    return pd.read_csv(p)


def by_police_service(csv_path: Path | str | None = None) -> RichResult:
    """Per-police-service tabulation: case counts, charges-recommended
    rate, common injuries.
    """
    df = _load(csv_path)
    g = df.groupby("police_service", dropna=False)
    n_cases = g.size()
    charges = g["charges_recommended"].apply(
        lambda s: s.eq(True).sum() if hasattr(s, "eq") else 0
    )
    no_charges = g["charges_recommended"].apply(
        lambda s: s.eq(False).sum() if hasattr(s, "eq") else 0
    )
    out_rows = []
    for svc in n_cases.index:
        n = int(n_cases[svc])
        c = int(charges.get(svc, 0))
        nc = int(no_charges.get(svc, 0))
        rate = (c / (c + nc)) if (c + nc) > 0 else float("nan")
        out_rows.append([str(svc)[:50], n, c, nc, f"{rate*100:.1f}%" if not pd.isna(rate) else "—"])
    out_rows.sort(key=lambda r: -r[1])  # by case count

    return RichResult(
        title="SIU cases by police service",
        summary_lines=[
            ("Unique police services", int(n_cases.size)),
            ("Total cases", int(df.shape[0])),
            ("With charges_recommended True", int(charges.sum())),
            ("With charges_recommended False", int(no_charges.sum())),
        ],
        tables=[{
            "title": "By police service (top 30):",
            "headers": ["Police service", "Cases", "Charged", "No charges", "Charge rate"],
            "rows": out_rows[:30],
        }],
        interpretation=(f"Top services by case count: "
                        f"{', '.join(r[0] for r in out_rows[:5])}. "
                        "Services with low charge rate may indicate either "
                        "truly justified force or systematic under-charging — "
                        "context-dependent interpretation."),
        payload={"counts": dict(zip([str(s) for s in n_cases.index], n_cases.tolist())),
                 "charges": charges.to_dict(),
                 "no_charges": no_charges.to_dict()},
    )


def by_year(csv_path: Path | str | None = None) -> RichResult:
    """Year-over-year case volume + charges rate from date_of_incident."""
    df = _load(csv_path)
    df["_year"] = df["date_of_incident_iso"].astype(str).str[:4]
    df["_year"] = pd.to_numeric(df["_year"], errors="coerce")
    valid = df.dropna(subset=["_year"])
    g = valid.groupby("_year")
    n = g.size()
    charged = g["charges_recommended"].apply(
        lambda s: s.eq(True).sum() if hasattr(s, "eq") else 0
    )
    no_charged = g["charges_recommended"].apply(
        lambda s: s.eq(False).sum() if hasattr(s, "eq") else 0
    )
    rows = []
    for year in sorted(n.index):
        rows.append([
            int(year), int(n[year]),
            int(charged.get(year, 0)),
            int(no_charged.get(year, 0)),
            (f"{100*charged[year]/(charged[year]+no_charged[year]):.1f}%"
             if (charged.get(year, 0) + no_charged.get(year, 0)) > 0 else "—"),
        ])
    return RichResult(
        title="SIU cases by year",
        summary_lines=[
            ("Years covered", f"{int(n.index.min())}–{int(n.index.max())}" if len(n) else "n/a"),
            ("Total cases with parseable date", int(valid.shape[0])),
            ("Cases with no parseable date", int(df.shape[0] - valid.shape[0])),
        ],
        tables=[{
            "title": "By year:",
            "headers": ["Year", "Cases", "Charged", "No charges", "Charge rate"],
            "rows": rows,
        }],
        payload={"by_year": {int(y): int(n[y]) for y in n.index}},
    )


def case_counts(csv_path: Path | str | None = None) -> RichResult:
    """Distribution of #SO, #WO, #CW per case."""
    df = _load(csv_path)
    rows = []
    for col, label in [
        ("number_of_subject_officials", "#Subject officials"),
        ("number_of_witness_officials", "#Witness officials"),
        ("number_of_civilian_witnesses", "#Civilian witnesses"),
        ("number_of_officers_involved", "#Officers involved"),
    ]:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.size == 0:
            rows.append([label, "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        rows.append([
            label, int(vals.size),
            f"{vals.mean():.2f}",
            f"{vals.median():.0f}",
            f"{vals.min():.0f}",
            f"{vals.max():.0f}",
        ])
    return RichResult(
        title="SIU case-team size distribution",
        summary_lines=[("Total cases", int(df.shape[0]))],
        tables=[{
            "title": "Per-case team / witness counts:",
            "headers": ["Field", "n parsed", "Mean", "Median", "Min", "Max"],
            "rows": rows,
        }],
    )


def demographics(csv_path: Path | str | None = None) -> RichResult:
    """Sex/age distribution of affected persons."""
    df = _load(csv_path)
    sex = df["sex_gender_affected"].fillna("unknown").value_counts()
    sex_rows = [[k, int(v), f"{100*v/sex.sum():.1f}%"] for k, v in sex.items()]
    age = pd.to_numeric(df["age_affected"], errors="coerce").dropna()
    return RichResult(
        title="Affected-person demographics",
        summary_lines=[
            ("Total cases", int(df.shape[0])),
            ("Cases with parseable age", int(age.size)),
            ("Mean age", float(age.mean()) if age.size else float("nan")),
            ("Median age", float(age.median()) if age.size else float("nan")),
            ("Age range", f"{int(age.min())}–{int(age.max())}" if age.size else "n/a"),
        ],
        tables=[{
            "title": "By sex/gender:",
            "headers": ["Sex/gender", "Count", "Percent"],
            "rows": sex_rows,
        }],
    )


def mental_health_race_indicators(csv_path: Path | str | None = None) -> RichResult:
    """Frequency of MH/race keyword indicators in narratives."""
    df = _load(csv_path)
    counts = Counter()
    nonempty = 0
    for sig in df["mental_health_or_race_indications"].dropna():
        s = str(sig).strip()
        if not s:
            continue
        nonempty += 1
        for kw in s.split(";"):
            kw = kw.strip()
            if kw:
                counts[kw] += 1
    rows = sorted(counts.items(), key=lambda kv: -kv[1])
    return RichResult(
        title="Mental-health / race indicators in SIU narratives",
        summary_lines=[
            ("Total cases", int(df.shape[0])),
            ("Cases with ≥1 indicator", nonempty),
            ("Distinct keywords matched", len(counts)),
        ],
        tables=[{
            "title": "Top keywords:",
            "headers": ["Keyword", "Cases mentioning"],
            "rows": [[k, v] for k, v in rows[:25]],
        }],
        warnings=[
            "Keyword-presence is a SIGNAL not a verdict. A case mentioning "
            "'mental health' may discuss it briefly without being primarily "
            "about MH. Read narratives in `SIU_narratives.jsonl` for context."
        ],
        interpretation=(
            f"{nonempty}/{int(df.shape[0])} cases ({100*nonempty/max(df.shape[0],1):.1f}%) "
            "have at least one MH or race keyword in the narrative. The "
            "distribution by keyword is shown above; see also `by_police_service` "
            "for service-by-service patterns."
        ),
    )


def decision_timing(csv_path: Path | str | None = None) -> RichResult:
    """Distributions of intervals: incident → notification → director's decision."""
    df = _load(csv_path)
    inc = pd.to_datetime(df["date_of_incident_iso"], errors="coerce")
    notif = pd.to_datetime(df["date_siu_notified_iso"], errors="coerce")
    decision = pd.to_datetime(df["date_of_director_decision_iso"], errors="coerce")

    inc_to_notif = (notif - inc).dt.days
    notif_to_decision = (decision - notif).dt.days
    inc_to_decision = (decision - inc).dt.days

    def _row(label, series):
        v = series.dropna()
        if v.size == 0:
            return [label, "n/a", "n/a", "n/a", "n/a", "n/a"]
        return [label, int(v.size), f"{v.mean():.1f}",
                f"{v.median():.0f}", f"{v.min():.0f}", f"{v.max():.0f}"]

    return RichResult(
        title="SIU decision timing (days)",
        summary_lines=[("Total cases", int(df.shape[0]))],
        tables=[{
            "title": "Interval distributions (days):",
            "headers": ["Interval", "n parsed", "Mean", "Median", "Min", "Max"],
            "rows": [
                _row("Incident → SIU notified", inc_to_notif),
                _row("Notified → director's decision", notif_to_decision),
                _row("Incident → decision (total)", inc_to_decision),
            ],
        }],
    )


def all_analyses(csv_path: Path | str | None = None,
                 out_dir: Path | None = None) -> dict:
    """Run every analysis and write each to its own file under
    data/manifest/outputs/siu/. Returns a dict of name → RichResult."""
    out_dir = out_dir or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    for name, fn in [
        ("by_police_service", by_police_service),
        ("by_year", by_year),
        ("case_counts", case_counts),
        ("demographics", demographics),
        ("mh_race_indicators", mental_health_race_indicators),
        ("decision_timing", decision_timing),
    ]:
        try:
            r = fn(csv_path)
            results[name] = r
            (out_dir / f"siu_analysis_{name}.txt").write_text(str(r))
            (out_dir / f"siu_analysis_{name}.json").write_text(
                json.dumps(r.payload, indent=2, default=str, ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            results[name] = RichResult(
                title=f"siu.{name} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
