"""moirais.doob_trends — National-aggregate analyses from Doob's Federal Court affidavit.

Replicates the analytical contribution of Prof. Anthony N. Doob's
expert-witness affidavit in *Canadian Civil Liberties Association et al.
v. The Attorney General of Canada* (Federal Court file T-539-20,
Application Record Vol. 3 of 5, pp. 778-795).

Doob's national-aggregate analyses (Figures 1-4 + Tables 1-3) sit
ALONGSIDE the per-row Ruhela formulations on OTIS provincial data and
the Doob chi-square family on aggregate contingency tables. Where
the Ruhela formulations test causal contrasts at the individual level
on Ontario provincial data, Doob's affidavit work tests *decoupling*
of imprisonment rates from crime rates at the Canadian and US
national-aggregate level over 50+ years.

Constants exposed
-----------------
  CCRSO_TABLE1_RELEASES      Table 1: 5-year average annual releases
                              by type (day / full parole / statutory)
                              with violent / non-violent / breach
                              revocations and successful completions.
  CCRSO_TABLE2_FLOW          Table 2: 2013/14-2017/18 prisoner flow:
                              count / admissions / deaths / full parole
                              releases / statutory releases.
  CCRSO_TABLE3_AGE           Table 3: 2018 age distribution — adult
                              population, CSC in-custody count, CSC
                              admissions — by age group (18-49, 50-59,
                              60+).

Functions exposed
-----------------
  analyze_doob_table1_releases() -> RichResult
      Renders Table 1 + computes overall success / revocation rates.
  analyze_doob_table2_flow() -> RichResult
      Renders Table 2 + year-over-year changes + 5-year averages.
  analyze_doob_table3_age_overrepresentation() -> RichResult
      Renders Table 3 + computes age-group IRR for CSC custody vs
      adult population (over- / under-representation).
  decoupling_test(crime_series, imprisonment_series) -> RichResult
      Tests Doob's thesis that imprisonment is decoupled from crime
      rate. Reports Pearson correlation between the two time series
      with optional Pettitt change-point detection.
  pettitt_changepoint(series) -> dict
      Pettitt non-parametric change-point detection for time series.

Data sources (cited in Doob affidavit)
--------------------------------------
  CCRSO 2018 — Public Safety Canada, Corrections and Conditional
    Release Statistical Overview 2018 (released 2019).
    https://www.publicsafety.gc.ca/cnt/rsrcs/pblctns/ccrso-2018/
  StatsCan CANSIM/Table 35-10-0026-01, 35-10-0177-01, 35-10-0014-01.
  Pastore, A. L. & Maguire, K. (2004). Sourcebook of Criminal Justice
    Statistics.

Citation
--------
Doob, A. N. (2020). Affidavit (T-539-20) of Anthony Doob — Federal
Court of Canada, Application Record Vol. 3 of 5. CCLA et al. v.
Attorney General of Canada.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .fn._richresult import RichResult


# ── Table 1: 5-year average annual releases (CCRSO 2013/14–2017/18) ─

CCRSO_TABLE1_RELEASES: list[dict] = [
    {
        "type": "Day Parole",
        "revoke_violent": 4.8, "revoke_violent_pct": 0.14,
        "revoke_non_violent": 33.8, "revoke_non_violent_pct": 1.0,
        "revoke_breach": 268.6, "revoke_breach_pct": 7.9,
        "success": 3085, "success_pct": 90.9,
        "total": 3392.2,
    },
    {
        "type": "Full Parole",
        "revoke_violent": 5.0, "revoke_violent_pct": 0.49,
        "revoke_non_violent": 28.2, "revoke_non_violent_pct": 2.8,
        "revoke_breach": 89.4, "revoke_breach_pct": 8.7,
        "success": 901.2, "success_pct": 88.0,
        "total": 1023.8,
    },
    {
        "type": "Statutory Release",
        "revoke_violent": 84.6, "revoke_violent_pct": 1.5,
        "revoke_non_violent": 452.8, "revoke_non_violent_pct": 7.8,
        "revoke_breach": 1556.0, "revoke_breach_pct": 26.7,
        "success": 3735.6, "success_pct": 64.1,
        "total": 5829.0,
    },
]


# ── Table 2: prisoner flow 2013/14–2017/18 (CCRSO) ─────────────────

CCRSO_TABLE2_FLOW: list[dict] = [
    {"year": "2013-14", "avg_count": 15342, "admissions": 5071,
     "deaths": 48, "full_parole_releases": 163, "statutory_releases": 5636},
    {"year": "2014-15", "avg_count": 14886, "admissions": 4818,
     "deaths": 67, "full_parole_releases": 185, "statutory_releases": 5373},
    {"year": "2015-16", "avg_count": 14712, "admissions": 4891,
     "deaths": 65, "full_parole_releases": 178, "statutory_releases": 5309},
    {"year": "2016-17", "avg_count": 14159, "admissions": 4908,
     "deaths": 47, "full_parole_releases": 166, "statutory_releases": 4888},
    {"year": "2017-18", "avg_count": 14092, "admissions": 4718,
     "deaths": None, "full_parole_releases": 208, "statutory_releases": 4427},
]


# ── Table 3: 2018 age distribution (Doob/CCRSO) ───────────────────

CCRSO_TABLE3_AGE: list[dict] = [
    {"age_group": "18-49",
     "canada_adult_pop": 15770626, "canada_adult_pop_pct": 52.8,
     "csc_in_custody": 10544, "csc_in_custody_pct": 74.8,
     "csc_admissions_2017_18": 3920, "csc_admissions_pct": 83.1},
    {"age_group": "50-59",
     "canada_adult_pop": 5305888, "canada_adult_pop_pct": 17.8,
     "csc_in_custody": 2236, "csc_in_custody_pct": 15.9,
     "csc_admissions_2017_18": 548, "csc_admissions_pct": 11.6},
    {"age_group": "60+",
     "canada_adult_pop": 8811720, "canada_adult_pop_pct": 29.5,
     "csc_in_custody": 1312, "csc_in_custody_pct": 9.3,
     "csc_admissions_2017_18": 250, "csc_admissions_pct": 5.3},
]


def analyze_doob_table1_releases() -> RichResult:
    """Doob Affidavit Table 1: 5-year average annual conditional releases."""
    rows = [
        [r["type"],
         f"{r['revoke_violent']:.1f} ({r['revoke_violent_pct']:.2f}%)",
         f"{r['revoke_non_violent']:.1f} ({r['revoke_non_violent_pct']:.1f}%)",
         f"{r['revoke_breach']:.1f} ({r['revoke_breach_pct']:.1f}%)",
         f"{r['success']:.1f} ({r['success_pct']:.1f}%)",
         f"{r['total']:.1f}"]
        for r in CCRSO_TABLE1_RELEASES
    ]
    total_pop = sum(r["total"] for r in CCRSO_TABLE1_RELEASES)
    total_violent_revokes = sum(r["revoke_violent"]
                                  for r in CCRSO_TABLE1_RELEASES)
    overall_violent_revoke_pct = (
        100.0 * total_violent_revokes / total_pop if total_pop else 0
    )
    return RichResult(
        title=("Doob Affidavit Table 1 — Successful & unsuccessful "
                "conditional releases (5-year avg, CCRSO 2013/14-2017/18)"),
        summary_lines=[
            ("Source", "CCRSO 2018 pp.94-98, Doob Affidavit Exhibit B"),
            ("Total annual releases (5-yr avg)", round(total_pop, 1)),
            ("Total annual revoke-violent (5-yr avg)",
                round(total_violent_revokes, 1)),
            ("Overall violent-revocation rate",
                f"{overall_violent_revoke_pct:.3f}%"),
            ("Doob's headline point",
                "≥ 99.4% of releases are successful or "
                "non-violent-revoked — the violent-revocation rate is < 1%"),
        ],
        tables=[{
            "title": ("Table 1: Average annual releases by type, "
                       "with revocation breakdowns:"),
            "headers": ["Type", "Revoke (violent)", "Revoke (non-violent)",
                         "Revoke (breach)", "Successful", "Total"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Doob's Federal Court Table 1. The headline "
            "finding: violent revocations are very rare — < 1% of "
            "all releases (across day parole, full parole, statutory) "
            "result in revocation due to a violent offence. The "
            "majority of unsuccessful releases are breach-of-condition "
            "(curfew, drug/alcohol restrictions, location restrictions), "
            "not new criminal conduct."
        ),
        payload={"table1": CCRSO_TABLE1_RELEASES,
                  "total_pop_per_year": total_pop,
                  "violent_revocation_rate_pct":
                    overall_violent_revoke_pct},
    )


def analyze_doob_table2_flow() -> RichResult:
    """Doob Affidavit Table 2: prisoner flow 2013/14-2017/18."""
    rows = [
        [r["year"], r["avg_count"], r["admissions"],
         (r["deaths"] if r["deaths"] is not None else "n/a"),
         r["full_parole_releases"], r["statutory_releases"]]
        for r in CCRSO_TABLE2_FLOW
    ]
    n_years = len(CCRSO_TABLE2_FLOW)
    avg_count = sum(r["avg_count"] for r in CCRSO_TABLE2_FLOW) / n_years
    avg_admissions = sum(r["admissions"]
                          for r in CCRSO_TABLE2_FLOW) / n_years
    avg_full_parole = sum(r["full_parole_releases"]
                           for r in CCRSO_TABLE2_FLOW) / n_years
    avg_stat_release = sum(r["statutory_releases"]
                            for r in CCRSO_TABLE2_FLOW) / n_years
    rows.append([
        "Average", round(avg_count, 1), round(avg_admissions, 1),
        45,  # Doob's stated 5-yr average
        round(avg_full_parole, 1), round(avg_stat_release, 1),
    ])
    monthly_releases = avg_stat_release / 12
    return RichResult(
        title=("Doob Affidavit Table 2 — Flow of prisoners into and out "
                "of penitentiaries (CCRSO 2013/14-2017/18)"),
        summary_lines=[
            ("Source", "CCRSO 2018, Doob Affidavit Exhibit B"),
            ("Average annual count (5-yr)", round(avg_count, 1)),
            ("Average annual admissions (5-yr)", round(avg_admissions, 1)),
            ("Average statutory releases / year (5-yr)",
                round(avg_stat_release, 1)),
            ("Average statutory releases / month",
                round(monthly_releases, 1)),
            ("Doob's headline point",
                "About 427 statutory releases per month — "
                "releasing prisoners 6 months early would empty "
                "~17.5% of the penitentiary system in one tranche."),
        ],
        tables=[{
            "title": "Table 2: Annual prisoner flow + 5-year average:",
            "headers": ["Year", "Avg count", "Admissions", "Deaths",
                         "Full parole rel.", "Statutory rel."],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Doob's Federal Court Table 2. The flow shows "
            "remarkable stability: counts within 14k–15k, admissions "
            "and statutory releases each ~5k/year. This stability is "
            "what enables Doob's projection that releasing prisoners "
            "6 months early would steady-state at ~427 fewer prisoners "
            "per month, depopulating ~17.5% of the system."
        ),
        payload={"table2": CCRSO_TABLE2_FLOW,
                  "avg_count": avg_count,
                  "avg_admissions": avg_admissions,
                  "monthly_releases": monthly_releases},
    )


def analyze_doob_table3_age_overrepresentation() -> RichResult:
    """Doob Affidavit Table 3: age distribution — Canada vs CSC custody."""
    rows = []
    irrs: list[dict] = []
    for r in CCRSO_TABLE3_AGE:
        # IRR for in-custody = (CSC pct / Canada pop pct)
        irr_custody = r["csc_in_custody_pct"] / r["canada_adult_pop_pct"]
        irr_admissions = r["csc_admissions_pct"] / r["canada_adult_pop_pct"]
        rows.append([
            r["age_group"],
            f"{r['canada_adult_pop']:,}",
            f"{r['canada_adult_pop_pct']:.1f}%",
            f"{r['csc_in_custody']:,}",
            f"{r['csc_in_custody_pct']:.1f}%",
            f"{r['csc_admissions_2017_18']:,}",
            f"{r['csc_admissions_pct']:.1f}%",
            f"{irr_custody:.2f}",
            f"{irr_admissions:.2f}",
        ])
        irrs.append({"age_group": r["age_group"],
                     "irr_custody": irr_custody,
                     "irr_admissions": irr_admissions})
    return RichResult(
        title=("Doob Affidavit Table 3 — Prisoner age distribution: "
                "Canada adult population vs CSC in-custody / admissions"),
        summary_lines=[
            ("Source", "CCRSO 2018 + StatsCan CANSIM, Doob Affidavit Exhibits A,C,D"),
            ("Age 50+ as % of Canada adult population", "47.3%"),
            ("Age 50+ as % of CSC in-custody", "25.2%"),
            ("Age 50+ as % of CSC admissions (2017-18)", "16.9%"),
            ("Doob's headline point",
                "Older adults are dramatically under-represented "
                "in CSC custody (25.2% vs 47.3%) and even more so in "
                "admissions (16.9%) — the prison population skews young."),
        ],
        tables=[{
            "title": ("Table 3: Age distribution + over-/under-"
                       "representation IRR (CSC%/Canada%):"),
            "headers": ["Age", "Adult pop", "Adult %", "In custody",
                         "Custody %", "Admissions", "Admissions %",
                         "IRR custody", "IRR admissions"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Doob's Federal Court Table 3 plus over-/under-"
            "representation IRRs. Age 18-49 has IRR_custody ≈ 1.42 "
            "(over-represented) and IRR_admissions ≈ 1.57; ages 50-59 "
            "near-balanced (IRR ~ 0.89/0.65); ages 60+ severely under-"
            "represented (IRR ~ 0.32/0.18). This supports Doob's "
            "argument that older prisoners are a low-risk group "
            "appropriate for conditional release."
        ),
        payload={"table3": CCRSO_TABLE3_AGE, "irrs": irrs},
    )


# ── Pettitt change-point detection ─────────────────────────────────


def pettitt_changepoint(series: "Iterable[float]") -> dict:
    """Pettitt's non-parametric change-point test (Pettitt 1979).

    Returns the location of the most likely single change-point in a
    time series, plus the test statistic and approximate p-value.

    >>> r = pettitt_changepoint([1, 1, 1, 1, 5, 5, 5, 5])
    >>> r["change_point_index"]
    3
    """
    arr = np.asarray(list(series), dtype=float)
    n = arr.size
    if n < 5:
        return {"change_point_index": None, "U_max": float("nan"),
                "p_value": float("nan"),
                "note": "n < 5; Pettitt test not applicable"}
    U = np.zeros(n)
    for t in range(1, n):
        a = arr[: t + 1].reshape(-1, 1)
        b = arr[t + 1:].reshape(1, -1)
        U[t] = float(np.sign(a - b).sum())
    abs_U = np.abs(U)
    k_max = int(abs_U.argmax())
    U_max = float(abs_U[k_max])
    # Pettitt approximate p-value (Pettitt 1979 eq. 11)
    p_approx = float(2.0 * np.exp(-6.0 * U_max ** 2 /
                                    (n ** 3 + n ** 2)))
    return {"change_point_index": k_max,
            "U_max": U_max,
            "p_value": min(p_approx, 1.0),
            "note": ("Pettitt 1979 approximate p-value; "
                      "single-change-point assumption")}


def decoupling_test(crime_series: "Iterable[float]",
                     imprisonment_series: "Iterable[float]",
                     years: "Iterable[int] | None" = None,
                     ) -> RichResult:
    """Doob's thesis test: imprisonment rate decoupled from crime rate.

    Computes Pearson correlation between the two equal-length time
    series. A correlation near 0 (or with a sign mismatch) supports
    Doob's claim that imprisonment is decoupled from crime trends.
    Optionally runs Pettitt change-point on each series.
    """
    crime = np.asarray(list(crime_series), dtype=float)
    imp = np.asarray(list(imprisonment_series), dtype=float)
    if crime.size != imp.size:
        return RichResult(
            title="Doob decoupling test",
            warnings=[f"length mismatch: crime={crime.size}, "
                       f"imp={imp.size}"],
        )
    if crime.size < 5:
        return RichResult(
            title="Doob decoupling test",
            warnings=[f"only {crime.size} years; need ≥ 5"],
        )
    r_pearson = float(np.corrcoef(crime, imp)[0, 1])
    n = crime.size
    # Fisher z approximate p-value for r != 0
    if abs(r_pearson) >= 1:
        p = 0.0
    else:
        z = 0.5 * np.log((1 + r_pearson) / (1 - r_pearson))
        from scipy import stats as sps
        p = float(2 * (1 - sps.norm.cdf(abs(z) * np.sqrt(n - 3))))
    # Pettitt on each
    pcp_crime = pettitt_changepoint(crime.tolist())
    pcp_imp = pettitt_changepoint(imp.tolist())
    yrs_str = (f"{int(min(years))}-{int(max(years))}"
                if years is not None else f"n={n} years")
    return RichResult(
        title=("Doob decoupling test — Pearson(crime rate, imprisonment "
                "rate) over time"),
        summary_lines=[
            ("Years", yrs_str),
            ("n", n),
            ("Pearson r", round(r_pearson, 4)),
            ("Two-sided p (Fisher z)", round(p, 6)),
            ("Pettitt change-point: crime",
                pcp_crime.get("change_point_index", "n/a")),
            ("Pettitt change-point: imprisonment",
                pcp_imp.get("change_point_index", "n/a")),
            ("Doob's thesis",
                "If r is small / sign-mismatched / non-significant, "
                "imprisonment is decoupled from crime — supports "
                "the affidavit's central claim."),
        ],
        interpretation=(
            "Tests Doob's central thesis from Federal Court Affidavit "
            "T-539-20 §§ 16-21: imprisonment rates and crime rates "
            "are not consistently correlated across time. r ≈ 0 with "
            "non-significant p ⇒ decoupling confirmed. r ≠ 0 with "
            "small magnitude ⇒ weak coupling. Pettitt change-points "
            "highlight whether either series has structural breaks."
        ),
        payload={"r_pearson": r_pearson, "p_value": p, "n": n,
                  "pettitt_crime": pcp_crime,
                  "pettitt_imp": pcp_imp},
    )


def analyze_doob_full_affidavit() -> RichResult:
    """Comprehensive replication of Doob's Federal Court affidavit
    aggregate analyses (Tables 1-3 + decoupling test stub).

    Renders Tables 1, 2, 3 + a placeholder for Figures 1-4 (which
    require StatsCan CANSIM time-series data not bundled with moirais).
    """
    t1 = analyze_doob_table1_releases()
    t2 = analyze_doob_table2_flow()
    t3 = analyze_doob_table3_age_overrepresentation()
    sections = []
    for label, r in [("§ Table 1", t1), ("§ Table 2", t2),
                      ("§ Table 3", t3)]:
        if r.tables:
            sections.append({
                "title": f"{label} — {r.title}",
                "headers": r.tables[0]["headers"],
                "rows": r.tables[0]["rows"],
            })
    return RichResult(
        title=("Doob Federal Court Affidavit (T-539-20) — aggregate "
                "national-level replication"),
        summary_lines=[
            ("Source", "Doob Affidavit T-539-20 Vol 3, pp. 778-795"),
            ("Tables replicated", "1, 2, 3"),
            ("Figures 1-4 (time series)",
                "require StatsCan CANSIM data; "
                "use decoupling_test(crime, imp) once series available"),
            ("Companion analyses in moirais",
                "Ruhela formulations on OTIS provincial data; "
                "Doob chi² on c/d-series; SIU IAP federal context"),
        ],
        tables=sections,
        interpretation=(
            "Replicates the three CCRSO tables Doob used to argue "
            "(a) violent revocation is rare (< 1% of releases), "
            "(b) prisoner flow is steady-state (~427 statutory "
            "releases/month), and (c) older adults are under-"
            "represented in CSC custody. These national-aggregate "
            "analyses sit alongside the Ruhela formulations on OTIS "
            "That which does not kill us makes us stronger. — Friedrich Nietzsche"
            "from federal aggregates down to provincial individual-"
            "level evidence."
        ),
        payload={
            "table1": CCRSO_TABLE1_RELEASES,
            "table2": CCRSO_TABLE2_FLOW,
            "table3": CCRSO_TABLE3_AGE,
        },
    )
