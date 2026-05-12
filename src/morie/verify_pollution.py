"""`morie verify-pollution` CLI handler (Workstream 6 of pollution-health plan).

One command wrapping `morie.envhealth` into a reproducible tool that
prints ATE / PAF / mortality-displaced / equity / assumption-log and
returns exit 0 on green, 1 on any assumption failure.

See ``.claude/plans/2026-04-17-pollution-health-mission.md`` §W6.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


# These imports are deferred until the command actually runs so that
# the runner's --help text works even when a user hasn't installed
# pandas / the envhealth deps.
def _load_deps():
    import numpy as np
    import pandas as pd

    from morie import envhealth
    return np, pd, envhealth


# -------------------- synthetic demo data --------------------

def _demo_exposure_and_outcome(pollutant: str):
    """Synthesize a plausible exposure + outcome dataset.

    Calibrated to Toronto FSA-level NO2 (Sentinel-5P) + CIHI asthma
    admissions orders of magnitude. Enough variation for the pipeline
    to return non-trivial effect sizes.
    """
    np, pd, _ = _load_deps()
    rng = np.random.default_rng(20260417)
    n = 1000
    if pollutant == "no2":
        exposure = rng.lognormal(mean=np.log(22), sigma=0.35, size=n)
    elif pollutant == "pm25":
        exposure = rng.lognormal(mean=np.log(9), sigma=0.30, size=n)
    else:
        exposure = rng.lognormal(mean=np.log(15), sigma=0.35, size=n)

    # Income quintile: 1 = lowest, 5 = highest. Numeric so
    # envhealth.pollution_equity_analysis can rank directly.
    income = rng.choice(
        [1, 2, 3, 4, 5], size=n,
        p=[0.18, 0.22, 0.22, 0.20, 0.18],
    )
    income_mult = {1: 1.25, 2: 1.12, 3: 1.00, 4: 0.93, 5: 0.85}
    exposure *= np.array([income_mult[int(q)] for q in income])

    return pd.DataFrame({
        "exposure": exposure,
        "income": income,
    })


# -------------------- assumption gates --------------------

def _check_assumptions(
    *,
    pollutant: str,
    exposure_mean: float,
    exposure_prevalence: float,
    baseline_rate: float,
    population: int,
    reference: float,
) -> list[dict[str, Any]]:
    """Return a list of {name, ok, note} rows for the assumption log."""
    assumptions: list[dict[str, Any]] = []

    def _add(name: str, ok: bool, note: str) -> None:
        assumptions.append({"assumption": name, "ok": bool(ok), "note": note})

    _add(
        "exposure > reference",
        exposure_mean > reference,
        f"mean {exposure_mean} vs ref {reference} — CRF is monotonic only "
        "when exposure exceeds the counterfactual floor.",
    )
    _add(
        "prevalence in [0,1]",
        0.0 <= exposure_prevalence <= 1.0,
        f"exposure_prevalence={exposure_prevalence}",
    )
    _add(
        "baseline_rate non-negative",
        baseline_rate >= 0.0,
        f"baseline_rate={baseline_rate} per 100k per year",
    )
    _add(
        "population positive",
        population > 0,
        f"population={population}",
    )
    _add(
        "pollutant supported by envhealth CRF",
        pollutant.lower() in ("no2", "pm25"),
        "Current CRFs: NO2 (log-linear), PM2.5 (Burnett IER). "
        "Other pollutants reject.",
    )
    return assumptions


# -------------------- main handler --------------------

def handle_verify_pollution(args: argparse.Namespace) -> int:
    """Dispatch for ``morie verify-pollution``.

    Returns 0 if all assumptions pass and the pipeline completes,
    1 if any assumption fails, 2 on file / data errors.
    """
    try:
        np, pd, envhealth = _load_deps()
    except ImportError as e:
        print(f"ERROR: missing dependency — {e}", file=sys.stderr)
        return 2

    pollutant = args.pollutant.lower()
    outcome = args.outcome
    reference = args.reference

    # --- Resolve exposure source ---
    equity_df = None
    if args.demo:
        df = _demo_exposure_and_outcome(pollutant)
        exposure_mean = float(df["exposure"].mean())
        exposure_prevalence = float((df["exposure"] > reference).mean())
        equity_df = df
        data_source = "demo (synthetic)"
    elif args.exposure_csv:
        p = Path(args.exposure_csv)
        if not p.exists():
            print(f"ERROR: exposure CSV not found: {p}", file=sys.stderr)
            return 2
        df = pd.read_csv(p)
        if "exposure" not in df.columns:
            print("ERROR: CSV missing 'exposure' column.", file=sys.stderr)
            return 2
        exposure_mean = float(df["exposure"].mean())
        exposure_prevalence = float((df["exposure"] > reference).mean())
        if "income" in df.columns:
            equity_df = df
        data_source = str(p)
    else:
        exposure_mean = float(args.exposure_mean)
        exposure_prevalence = float(args.exposure_prevalence)
        data_source = "CLI scalar args"

    baseline_rate = float(args.baseline_rate)
    population = int(args.population)

    # --- Check assumptions ---
    assumptions = _check_assumptions(
        pollutant=pollutant,
        exposure_mean=exposure_mean,
        exposure_prevalence=exposure_prevalence,
        baseline_rate=baseline_rate,
        population=population,
        reference=reference,
    )
    failed = [a for a in assumptions if not a["ok"]]

    report: dict[str, Any] = {
        "command": "morie verify-pollution",
        "pollutant": pollutant,
        "outcome": outcome,
        "region": args.region,
        "years": args.years,
        "data_source": data_source,
        "inputs": {
            "exposure_mean": exposure_mean,
            "exposure_prevalence": exposure_prevalence,
            "baseline_rate_per_100k": baseline_rate,
            "population": population,
            "reference_conc": reference,
        },
        "assumptions": assumptions,
    }

    # Short-circuit if any assumption failed (and exit 1)
    if failed:
        report["status"] = "assumption_failure"
        report["pipeline"] = {"skipped": True, "reason": "assumptions"}
        _emit(report, as_json=args.json)
        return 1

    # --- Pipeline stage 1: concentration-response ---
    if pollutant == "no2":
        crf = envhealth.concentration_response_no2(
            exposure_mean, outcome=outcome, reference_conc=reference,
        )
    else:  # pm25
        crf = envhealth.concentration_response_pm25(
            exposure_mean, outcome=outcome, reference_conc=reference,
        )

    # --- Stage 2: attributable fraction ---
    paf = envhealth.attributable_fraction(
        rr=float(crf.rr),
        exposure_prevalence=exposure_prevalence,
    )

    # --- Stage 3: mortality displaced ---
    # envhealth expects baseline_rate as deaths per person-year (e.g. 0.005).
    # The CLI takes "per 100,000 per year" for readability; convert here.
    baseline_rate_per_person = baseline_rate / 100_000.0
    exposure_delta = max(0.0, exposure_mean - reference)
    beta_per_unit = float(np.log(crf.rr) / max(exposure_delta, 1e-9))
    displaced_n = envhealth.mortality_displaced(
        exposure_delta=exposure_delta,
        population=population,
        baseline_rate=baseline_rate_per_person,
        beta_per_unit=beta_per_unit,
    )

    # --- Stage 4: end-to-end burden ---
    # envhealth expects 'PM2.5' (with decimal) and 'NO2' (uppercase)
    burden_name = "PM2.5" if pollutant == "pm25" else "NO2"
    burden = envhealth.burden_of_pollution(
        exposure_mean=exposure_mean,
        exposure_prevalence=exposure_prevalence,
        baseline_rate=baseline_rate_per_person,
        population=population,
        pollutant=burden_name,
    )

    # --- Stage 5: equity (only if we have demographic data) ---
    equity: Any = None
    if equity_df is not None and "income" in equity_df.columns:
        equity = envhealth.pollution_equity_analysis(
            equity_df, exposure="exposure", income="income",
        )

    report["status"] = "ok"
    report["pipeline"] = {
        "crf":       _dump(crf),
        "paf":       paf,
        "displaced": {
            "deaths_displaced":  float(displaced_n),
            "exposure_delta":    exposure_delta,
            "beta_per_unit":     beta_per_unit,
        },
        "burden":    _dump(burden),
        "equity":    _dump(equity) if equity is not None else None,
    }

    _emit(report, as_json=args.json)
    return 0


# -------------------- output helpers --------------------

def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def _emit(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, default=str))
        return

    print("=" * 66)
    print(f"  morie verify-pollution -- {report['pollutant'].upper()}"
          f" -> {report['outcome']}")
    if report.get("region"):
        print(f"  region: {report['region']}")
    if report.get("years"):
        print(f"  years:  {report['years']}")
    print(f"  data:   {report['data_source']}")
    print("=" * 66)

    inp = report["inputs"]
    print("\nInputs")
    print(f"  exposure mean:      {inp['exposure_mean']:.3f}")
    print(f"  exposure prevalence:{inp['exposure_prevalence']:.3f}")
    print(f"  baseline rate/100k: {inp['baseline_rate_per_100k']:.2f}")
    print(f"  population:         {inp['population']:,}")
    print(f"  reference conc:     {inp['reference_conc']}")

    print("\nAssumption log")
    for a in report["assumptions"]:
        tick = "PASS" if a["ok"] else "FAIL"
        print(f"  [{tick}] {a['assumption']} — {a['note']}")

    if report["status"] == "assumption_failure":
        print("\nSTATUS: assumption_failure (pipeline skipped)")
        return

    pipe = report["pipeline"]
    crf = pipe["crf"]
    print("\nConcentration-response")
    print(f"  RR:       {crf['rr']:.4f}")
    ci = crf.get("ci95") or crf.get("ci")
    if ci:
        print(f"  95% CI:   ({ci[0]:.4f}, {ci[1]:.4f})")
    print(f"  source:   {crf.get('source', '?')}")

    print(f"\nAttributable fraction (PAF): {pipe['paf']:.4f}")

    d = pipe["displaced"]
    print("\nMortality displaced")
    print(f"  expected avoided deaths: {d.get('deaths_displaced', 0):.1f}")

    b = pipe["burden"]
    print("\nBurden of pollution")
    print(f"  attributable deaths:   {b.get('deaths_attributable', 0):.1f}")
    if "dalys" in b and b["dalys"] is not None:
        print(f"  DALYs:                 {b['dalys']:.1f}")

    if pipe.get("equity") is not None:
        eq = pipe["equity"]
        print("\nEquity analysis")
        print(f"  concentration index (Gini): {eq.get('concentration_index', 0):.4f}")
        if "ratio_Q1_to_Q5" in eq:
            print(f"  exposure ratio Q1/Q5:       {eq['ratio_Q1_to_Q5']:.3f}")

    print("\nSTATUS: ok")


# -------------------- argparse registration --------------------

def register_subparser(subparsers) -> None:
    """Called from morie.runner.build_parser() to register this command."""
    p = subparsers.add_parser(
        "verify-pollution",
        help="Run a pollution → health causal pipeline and print a report.",
    )
    p.add_argument("--pollutant", required=True,
                    choices=["no2", "pm25", "NO2", "PM25"],
                    help="Pollutant to analyze.")
    p.add_argument("--outcome", default="all_cause_mortality",
                    help="Outcome name passed to concentration_response_*.")
    p.add_argument("--region", default=None,
                    help="Region label for reporting (e.g. ON-FSA-M6H).")
    p.add_argument("--years", default=None,
                    help="Year range label for reporting (e.g. 2019-2023).")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--demo", action="store_true",
                     help="Use synthetic demo data.")
    src.add_argument("--exposure-csv", default=None,
                     help="CSV with 'exposure' (+ optional 'income') columns.")

    p.add_argument("--exposure-mean", type=float, default=0.0,
                    help="Scalar exposure mean (µg/m³). Used if no csv/demo.")
    p.add_argument("--exposure-prevalence", type=float, default=0.0,
                    help="Fraction of population above reference.")
    p.add_argument("--reference", type=float, default=5.8,
                    help="Counterfactual reference concentration (µg/m³).")
    p.add_argument("--baseline-rate", type=float, default=500.0,
                    help="Baseline outcome rate per 100,000 per year.")
    p.add_argument("--population", type=int, default=1_000_000,
                    help="Population at risk.")
    p.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the text report.")
