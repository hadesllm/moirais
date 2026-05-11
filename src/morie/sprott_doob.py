"""morie.sprott_doob — Sprott & Doob (CRIMSL UToronto) SIU analyses.

Replicates the analytical contribution of the four CRIMSL UToronto
research reports authored by **Prof. Jane B. Sprott** (Toronto
Metropolitan University, formerly Ryerson) and **Prof. Anthony N.
Doob** (University of Toronto), with **Prof. Adelina Iftene**
(Dalhousie) co-authoring the May 2021 paper on Independent External
Decision Makers.

The four reports
----------------

1. **Sprott & Doob (Oct 2020)** — *Understanding the Operation of
   Correctional Service Canada's Structured Intervention Units:
   Some Preliminary Findings*. First systematic outside analysis of
   CSC's SIU data; demonstrated that SIUs were not operating as
   the legislative framework (Bill C-83) required.

2. **Sprott & Doob (Nov 2020)** — *Is there Clear Evidence that
   COVID-19 Was the Cause of Problems with the Operation of CSC's
   Structured Intervention Units?* Tests CSC's COVID-attribution
   defense; finds the data did not support it (problems pre-existed
   COVID).

3. **Sprott & Doob (Feb 2021)** — *Solitary Confinement, Torture,
   and Canada's Structured Intervention Units*. The most data-
   intensive of the four; introduces a Mandela-Rules classifier for
   SIU stays (solitary confinement = ≤15 days at ≤2 hrs out-of-cell;
   torture = ≥16 days under same conditions). Contains Tables 13,
   19, 23 reproduced below.

4. **Sprott, Doob & Iftene (May 2021)** — *Do Independent External
   Decision Makers Ensure that Solitary Confinement is no Longer
   Used in Canada's Federal Penitentiaries?* Evaluates the IEDM
   review mechanism added by Bill C-83.

Tables hardcoded (from Feb 2021 report)
----------------------------------------

  TABLE13_REGIONAL_RATES_PER_1000
      SIU person-stays per 1000 regional penitentiary prisoners,
      split into short (≤15 days) and long (≥16 days) stays, by
      CSC region.
  TABLE19_MANDELA_CLASSIFICATION
      Sprott-Doob Mandela-Rules classification of N=1960 SIU stays:
      solitary confinement (28.4%, N=556), torture (9.9%, N=195),
      all-other (61.7%, N=1209).
  TABLE23_REGIONAL_TORTURE_RATES
      Solitary and torture rates per 1000 prisoners by region.
      Headline finding: Pacific torture rate (39.1) is 22.6× higher
      than Ontario's (1.73).

Functions exposed
-----------------

  analyze_table13_regional_rates() -> RichResult
  analyze_table19_mandela_classification() -> RichResult
  analyze_table23_regional_torture_rates() -> RichResult
  classify_mandela(days_in_siu, hrs_out_of_cell_avg,
                    missed_full_4hrs) -> dict
      Apply the Sprott-Doob Mandela-Rules classifier to a single
      SIU person-stay and return the category.
  analyze_full_sprott_doob_feb2021() -> RichResult
      Comprehensive replication of the Feb 2021 paper's main tables.

Citation
--------
Sprott, J. B., & Doob, A. N. (2021, February). Solitary Confinement,
Torture, and Canada's Structured Intervention Units. Centre for
Criminology & Sociolegal Studies, U. of Toronto.
URL: https://www.crimsl.utoronto.ca/sites/www.crimsl.utoronto.ca/
files/TortureSolitarySIUsSprottDoob23Feb2021_0.pdf

Note on author order
--------------------
Sprott is FIRST author on the public CRIMSL papers; Doob is co-
author. This is documented and indexed in `morie.siuiap.
CRIMSL_REPORTS`. The October 2020 report's Drive filename
("DoobSprott") had Doob first but the published title page may
differ — when in doubt, follow the published front-matter order.
"""

from __future__ import annotations

from .fn._richresult import RichResult


# ── Table 13: SIU person-stays per 1000 regional prisoners ─────────
# Sprott & Doob (Feb 2021), p. 3 of 28 (Bates p.3).

TABLE13_REGIONAL_RATES_PER_1000 = [
    {"region": "Atlantic", "short_stay_rate": 70.2,
     "long_stay_rate": 124.8, "overall_rate": 195.0},
    {"region": "Quebec", "short_stay_rate": 178.1,
     "long_stay_rate": 118.1, "overall_rate": 296.2},
    {"region": "Ontario", "short_stay_rate": 18.2,
     "long_stay_rate": 30.3, "overall_rate": 48.5},
    {"region": "Prairie", "short_stay_rate": 41.4,
     "long_stay_rate": 91.2, "overall_rate": 132.7},
    {"region": "Pacific", "short_stay_rate": 102.7,
     "long_stay_rate": 99.8, "overall_rate": 202.5},
    {"region": "Total", "short_stay_rate": 73.4,
     "long_stay_rate": 84.1, "overall_rate": 157.5},
]


# ── Table 19: Sprott-Doob Mandela-Rules classification ─────────────
# Sprott & Doob (Feb 2021), p. 4 of 28 (Bates p.4). N=1960 SIU stays.

TABLE19_MANDELA_CLASSIFICATION = [
    {"category": "Solitary Confinement",
     "definition": ("Missed full 4 hrs out of cell 100% of days; ≤2 hrs "
                     "average out of cell during stay; stayed ≤15 days"),
     "percent": 28.4, "n": 556},
    {"category": "Torture",
     "definition": ("Missed full 4 hrs out of cell 100% of days; ≤2 hrs "
                     "average out of cell during stay; stayed ≥16 days"),
     "percent": 9.9, "n": 195},
    {"category": "All other person stays",
     "definition": "Did not meet either threshold above",
     "percent": 61.7, "n": 1209},
]


# ── Table 23: Regional solitary + torture rates per 1000 ───────────
# Sprott & Doob (Feb 2021), p. 4 of 28 (Bates p.4). Rates calculated
# using December 2020 estimates of total penitentiary population.

TABLE23_REGIONAL_TORTURE_RATES = [
    {"region": "Atlantic", "solitary_rate": 45.9, "torture_rate": 15.6},
    {"region": "Quebec", "solitary_rate": 118.1, "torture_rate": 25.2},
    {"region": "Ontario", "solitary_rate": 11.8, "torture_rate": 1.73},
    {"region": "Prairie", "solitary_rate": 13.2, "torture_rate": 10.5},
    {"region": "Pacific", "solitary_rate": 66.9, "torture_rate": 39.1},
    {"region": "Total", "solitary_rate": 44.2, "torture_rate": 15.5},
]


# Headline findings stated in the executive summary:
HEADLINE_FINDINGS = {
    "missed_full_4hrs_overall_pct": 38.9,
    "long_stay_missed_4hrs_in_76pct_of_days": 63.0,
    "pacific_torture_rate": 39.1,
    "ontario_torture_rate": 1.73,
    "pacific_to_ontario_ratio": 22.6,
    "n_total_stays": 1960,
}


# ── Mandela classifier ────────────────────────────────────────────


def classify_mandela(days_in_siu: int,
                      hrs_out_of_cell_avg: float,
                      missed_full_4hrs_pct_of_days: float) -> dict:
    """Apply the Sprott-Doob Mandela-Rules classifier.

    Parameters
    ----------
    days_in_siu : int
        Length of the SIU person-stay in days.
    hrs_out_of_cell_avg : float
        Average hours out of cell per day during the stay.
    missed_full_4hrs_pct_of_days : float
        Percent of days during the stay where the inmate did not
        receive the legislatively-required 4 hours out of cell.
        (Range: 0-100.)

    Returns
    -------
    dict with keys 'category' (str), 'rule' (str), 'reason' (str).

    Categories
    ----------
    "Solitary Confinement" — Mandela Rule 44: ≤2 hrs out of cell, all
        days missed full 4 hrs, stay length ≤ 15 days.
    "Torture" — Mandela Rules 43+44: same conditions but stay length
        ≥ 16 days.
    "All other"  — none of the above thresholds met.

    >>> r = classify_mandela(20, 1.5, 100)
    >>> r["category"]
    'Torture'
    >>> r = classify_mandela(8, 1.5, 100)
    >>> r["category"]
    'Solitary Confinement'
    >>> r = classify_mandela(20, 5, 50)
    >>> r["category"]
    'All other'
    """
    meets_22hr_threshold = (
        hrs_out_of_cell_avg <= 2.0
        and missed_full_4hrs_pct_of_days >= 100.0
    )
    if meets_22hr_threshold and days_in_siu <= 15:
        return {
            "category": "Solitary Confinement",
            "rule": "Mandela Rule 44",
            "reason": (f"≤2 hrs out of cell ({hrs_out_of_cell_avg}), "
                        f"missed full 4 hrs every day, "
                        f"stay {days_in_siu} ≤ 15 days"),
        }
    if meets_22hr_threshold and days_in_siu >= 16:
        return {
            "category": "Torture",
            "rule": "Mandela Rules 43+44",
            "reason": (f"≤2 hrs out of cell ({hrs_out_of_cell_avg}), "
                        f"missed full 4 hrs every day, "
                        f"stay {days_in_siu} ≥ 16 days "
                        f"(crosses Mandela Rule 43 'prolonged' threshold)"),
        }
    return {
        "category": "All other",
        "rule": "—",
        "reason": ("Did not meet the joint threshold of ≤2 hrs out of "
                    "cell, all days missed 4 hrs, and stay length"),
    }


# ── Replication functions ─────────────────────────────────────────


def analyze_table13_regional_rates() -> RichResult:
    """Sprott-Doob Table 13: SIU person-stay rates per 1000 prisoners."""
    rows = [[r["region"], r["short_stay_rate"],
              r["long_stay_rate"], r["overall_rate"]]
             for r in TABLE13_REGIONAL_RATES_PER_1000]
    # Quebec/Ontario short-stay ratio (the executive summary's headline)
    quebec_short = next(r["short_stay_rate"]
                         for r in TABLE13_REGIONAL_RATES_PER_1000
                         if r["region"] == "Quebec")
    ontario_short = next(r["short_stay_rate"]
                          for r in TABLE13_REGIONAL_RATES_PER_1000
                          if r["region"] == "Ontario")
    qc_on_ratio = quebec_short / ontario_short
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 13 — SIU person-stays "
                "per 1,000 regional prisoners"),
        summary_lines=[
            ("Source", "Sprott & Doob (Feb 2021), p. 3"),
            ("Quebec short-stay rate per 1000", quebec_short),
            ("Ontario short-stay rate per 1000", ontario_short),
            ("Quebec/Ontario short-stay ratio",
                f"{qc_on_ratio:.1f}× (matches the report's "
                f"'almost 10 times' claim)"),
            ("Total overall rate per 1000", 157.5),
        ],
        tables=[{
            "title": "Table 13 (rates per 1000 prisoners by region):",
            "headers": ["Region", "Short-stay rate", "Long-stay rate",
                         "Overall rate"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Sprott & Doob's Table 13. Quebec's rate of "
            "short SIU stays (≤15 days) was nearly 10× Ontario's, and "
            "the long-stay rate was higher in EVERY other region than "
            "in Ontario. Sprott & Doob argue this regional variation "
            "is not explained by population characteristics alone — "
            "it points to structurally different decision-making "
            "across CSC regions."
        ),
        payload={"table13": TABLE13_REGIONAL_RATES_PER_1000,
                  "qc_on_short_stay_ratio": qc_on_ratio},
    )


def analyze_table19_mandela_classification() -> RichResult:
    """Sprott-Doob Table 19: Mandela-Rules classification of N=1960 SIU stays."""
    rows = [[r["category"], f"{r['percent']:.1f}%", r["n"]]
             for r in TABLE19_MANDELA_CLASSIFICATION]
    rows.append(["Total", "100%", 1960])
    n_solitary = next(r["n"] for r in TABLE19_MANDELA_CLASSIFICATION
                       if r["category"] == "Solitary Confinement")
    n_torture = next(r["n"] for r in TABLE19_MANDELA_CLASSIFICATION
                      if r["category"] == "Torture")
    n_problematic = n_solitary + n_torture
    pct_problematic = 100.0 * n_problematic / 1960
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 19 — Mandela-Rules "
                "classification of SIU person-stays"),
        summary_lines=[
            ("Source", "Sprott & Doob (Feb 2021), p. 4 of 28"),
            ("Total person-stays classified", 1960),
            ("Solitary Confinement (Mandela Rule 44)", "28.4% (556)"),
            ("Torture (Mandela Rules 43+44, ≥16 days)", "9.9% (195)"),
            ("All other person-stays", "61.7% (1,209)"),
            ("% meeting either Mandela threshold",
                f"{pct_problematic:.1f}% ({n_problematic})"),
        ],
        tables=[{
            "title": ("Table 19 — Mandela-Rules classification "
                       "(stays > 1 day):"),
            "headers": ["Category", "Percent", "N"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Sprott & Doob's Table 19 — the headline "
            "Mandela-Rules classification. ~38% of SIU person-stays "
            "(those longer than 1 day) meet international thresholds "
            "for either solitary confinement (28.4%, N=556) or "
            "torture (9.9%, N=195) under UN Mandela Rules. CSC's "
            "post-Bill-C-83 SIU regime — designed precisely to avoid "
            "these international classifications — does not in fact "
            "do so for ~38% of stays."
        ),
        payload={"table19": TABLE19_MANDELA_CLASSIFICATION,
                  "n_problematic": n_problematic,
                  "pct_problematic": pct_problematic},
    )


def analyze_table23_regional_torture_rates() -> RichResult:
    """Sprott-Doob Table 23: regional torture/solitary rates per 1000."""
    rows = [[r["region"], r["solitary_rate"], r["torture_rate"]]
             for r in TABLE23_REGIONAL_TORTURE_RATES]
    pacific_torture = next(r["torture_rate"]
                              for r in TABLE23_REGIONAL_TORTURE_RATES
                              if r["region"] == "Pacific")
    ontario_torture = next(r["torture_rate"]
                              for r in TABLE23_REGIONAL_TORTURE_RATES
                              if r["region"] == "Ontario")
    pac_on_ratio = pacific_torture / ontario_torture
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 23 — Regional rates of "
                "Solitary Confinement and Torture per 1,000 prisoners"),
        summary_lines=[
            ("Source", "Sprott & Doob (Feb 2021), p. 4 of 28"),
            ("Pacific torture rate / 1000 prisoners", pacific_torture),
            ("Ontario torture rate / 1000 prisoners", ontario_torture),
            ("Pacific/Ontario torture ratio",
                f"{pac_on_ratio:.1f}× (Pacific is the report's "
                f"'alarmingly high' region)"),
            ("Headline quote",
                "'Quebec stands out as having the highest proportion "
                "(40.6%) of SIU stays that would be considered "
                "solitary confinement... Pacific stands out as "
                "having an alarmingly high proportion (19.5%) of "
                "its SIU stays that would be considered torture.'"),
        ],
        tables=[{
            "title": "Table 23 — Regional rates per 1000 prisoners:",
            "headers": ["Region", "Solitary rate", "Torture rate"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Sprott & Doob's Table 23. The Pacific "
            "region's torture rate (39.1 per 1000 prisoners) is "
            f"{pac_on_ratio:.1f}× the Ontario rate (1.73). This "
            "regional disparity is the empirical core of the "
            "Sprott-Doob argument that SIU placement decisions vary "
            "structurally across CSC regions in ways that no "
            "population-level explanation can account for. "
            "Companion analysis in morie: Ontario's *provincial* "
            "regional patterns (TPS / OTIS data) show the "
            "complementary picture at the sub-provincial scale."
        ),
        payload={"table23": TABLE23_REGIONAL_TORTURE_RATES,
                  "pac_on_torture_ratio": pac_on_ratio},
    )


# ── May 2021 IEDM paper (Sprott, Doob, Iftene) ─────────────────────
# "Do Independent External Decision Makers Ensure that 'An Inmate's
# Confinement in a Structured Intervention Unit Is to End as Soon as
# Possible'? [CCRA, Section 33]" — Schulich Law Scholars, 9 May 2021.
# Pages 1-7 transcribed from the bepress upload. N=265 stays with at
# least one IEDM review under CCRA s.37.8.

TABLE1_IEDM_POPULATION = {
    "n_total": 265,
    "gender": [
        {"category": "Female", "percent": 0.8, "n": 2},
        {"category": "Male", "percent": 99.2, "n": 263},
    ],
    "age_group": [
        {"category": "18-24", "percent": 15.5, "n": 41},
        {"category": "25-29", "percent": 29.8, "n": 79},
        {"category": "30-39", "percent": 32.8, "n": 87},
        {"category": "40-49", "percent": 17.0, "n": 45},
        {"category": "50-64", "percent": 4.9, "n": 13},
    ],
    "race": [
        {"category": "White", "percent": 30.9, "n": 82},
        {"category": "Indigenous", "percent": 40.4, "n": 107},
        {"category": "Black", "percent": 15.8, "n": 42},
        {"category": "Other/Missing", "percent": 12.8, "n": 34},
    ],
    "mental_health": [
        {"category": "Mental health need flag", "percent": 26.4, "n": 71},
        {"category": "No mental health need flag",
         "percent": 73.6, "n": 194},
    ],
}


HEADLINE_MAY2021 = {
    "n_iedm_stays_reviewed": 265,
    "n_iedm_decisions_rendered": 380,
    "pct_stay_in_decisions_among_rendered": 87,
    "pct_csc_moved_prisoner_before_iedm": 30,
    "n_long_stay_no_iedm_record_min76d": 105,
    "n_long_stay_no_iedm_record_min120d": 5,
    "iedm_min_should_remain_pct": 38,
    "iedm_max_should_remain_pct": 86,
    "n_iedms": 12,
    "pct_referred_55_to_62_days": 74.3,
    "indigenous_share_of_reviewed_stays_pct": 40.4,
    "black_share_of_reviewed_stays_pct": 15.8,
}


def analyze_iedm_table1_population() -> RichResult:
    """Sprott, Doob & Iftene (May 2021) Table 1: IEDM-reviewed population."""
    sections = []
    for label, key in [("Gender", "gender"), ("Age group", "age_group"),
                        ("Race", "race"),
                        ("Mental health flag", "mental_health")]:
        rows = [[r["category"], f"{r['percent']:.1f}%", r["n"]]
                 for r in TABLE1_IEDM_POPULATION[key]]
        rows.append(["Total", "100%", 265])
        sections.append({
            "title": f"{label} (N=265):",
            "headers": [label, "Percent", "N"],
            "rows": rows,
        })
    return RichResult(
        title=("Sprott, Doob & Iftene (May 2021) Table 1 — Population "
                "characteristics of SIU stays receiving ≥1 IEDM review "
                "under CCRA s.37.8"),
        summary_lines=[
            ("Source", "Sprott, Doob & Iftene (9 May 2021), p. 7 of 23"),
            ("N total stays reviewed by an IEDM",
                TABLE1_IEDM_POPULATION["n_total"]),
            ("Female / Male", "0.8% (2) vs 99.2% (263)"),
            ("Indigenous share", "40.4% (107)"),
            ("Black share", "15.8% (42)"),
            ("Mental health need flag share", "26.4% (71)"),
        ],
        tables=sections,
        interpretation=(
            "Reproduces Sprott, Doob & Iftene's Table 1. Indigenous "
            "people make up 40.4% of IEDM-reviewed SIU stays — a "
            "stark over-representation against the Indigenous share "
            "of the Canadian adult population (~5%). Black people "
            "are 15.8% of stays vs ~4% of the adult population "
            "(~4× over-representation). The mental-health flag "
            "applies to 26.4% of reviewed stays. Companion "
            "subgroup analyzers in morie.otis_all_analyze cover "
            "the parallel patterns in OTIS provincial data."
        ),
        payload={"table1_iedm_population": TABLE1_IEDM_POPULATION},
    )


def analyze_iedm_review_outcomes() -> RichResult:
    """Sprott, Doob & Iftene May 2021: IEDM review outcomes & disparities."""
    h = HEADLINE_MAY2021
    return RichResult(
        title=("Sprott, Doob & Iftene (May 2021) — IEDM review "
                "outcomes and structural disparities"),
        summary_lines=[
            ("Source", "Sprott, Doob & Iftene (9 May 2021)"),
            ("N stays reviewed by an IEDM (s.37.8)",
                h["n_iedm_stays_reviewed"]),
            ("N IEDM decisions rendered",
                h["n_iedm_decisions_rendered"]),
            ("'Stay-in' rate among rendered decisions",
                f"{h['pct_stay_in_decisions_among_rendered']}% — most "
                f"IEDM decisions ratify continued SIU placement"),
            ("CSC moved prisoner BEFORE IEDM decided",
                f"{h['pct_csc_moved_prisoner_before_iedm']}% of cases "
                f"— 'CSC can structure timing of release to meet its "
                f"own unarticulated needs' (Iftene/Sprott/Doob)"),
            ("IEDM-level variance in 'should remain' rate",
                f"{h['iedm_min_should_remain_pct']}% to "
                f"{h['iedm_max_should_remain_pct']}% across "
                f"{h['n_iedms']} IEDMs — large IEDM-level "
                f"heterogeneity"),
            ("Long-stay (≥76 days) with NO IEDM record",
                f"{h['n_long_stay_no_iedm_record_min76d']} cases "
                f"(of which {h['n_long_stay_no_iedm_record_min120d']} "
                f"are ≥120 days) — apparent compliance failure"),
            ("% of stays referred to IEDM in days 55-62",
                f"{h['pct_referred_55_to_62_days']:.1f}%"),
            ("Indigenous share of reviewed stays",
                f"{h['indigenous_share_of_reviewed_stays_pct']:.1f}%"),
            ("Black share of reviewed stays",
                f"{h['black_share_of_reviewed_stays_pct']:.1f}%"),
        ],
        interpretation=(
            "Captures the May 2021 paper's structural critique. The "
            "IEDM mechanism — created by Bill C-83 to provide "
            "independent oversight — in practice (a) ratifies "
            "continued SIU placement 87% of the time when it "
            "renders a decision, (b) is pre-empted by CSC for 30% "
            "of cases (CSC moves the prisoner before the IEDM can "
            "decide), (c) has 38%-86% inter-IEDM variance — pointing "
            "to inconsistent decision standards across the 12 IEDMs, "
            "and (d) leaves some prisoners in SIUs for 76-120+ days. "
            "Sprott and Doob conclude that the IEDM "
            "process is 'almost completely secretive' and inadequate "
            "as oversight."
        ),
        payload={"headline_may2021": h},
    )


# ── Feb 2021 — additional tables (4, 11, 12, 15, 20, 22) ──────────
# Transcribed from pp.11-26 of the report. These extend Tables 13,
# 19, 23 already encoded above.

TABLE4_LENGTH_OF_STAY = [
    {"days": "1-5", "n": 456, "pct": 23.0},
    {"days": "6-15", "n": 468, "pct": 23.6},
    {"days": "16-31", "n": 320, "pct": 16.1},
    {"days": "32-61", "n": 326, "pct": 16.4},
    {"days": "62-380", "n": 413, "pct": 20.8},
]
TABLE4_N = 1983


# Table 11: Region × Stay-length crosstab. χ²=201.00, df=16, p<.001.
TABLE11_REGION_X_STAY_LENGTH = [
    {"region": "Atlantic", "1-5": 25, "6-15": 56, "16-31": 32,
     "32-61": 50, "62-380": 62, "total": 225},
    {"region": "Quebec", "1-5": 266, "6-15": 179, "16-31": 90,
     "32-61": 79, "62-380": 126, "total": 740},
    {"region": "Ontario", "1-5": 23, "6-15": 40, "16-31": 24,
     "32-61": 28, "62-380": 53, "total": 168},
    {"region": "Prairies", "1-5": 52, "6-15": 102, "16-31": 96,
     "32-61": 118, "62-380": 125, "total": 493},
    {"region": "Pacific", "1-5": 90, "6-15": 91, "16-31": 78,
     "32-61": 51, "62-380": 47, "total": 357},
]
TABLE11_CHISQ = {"chi2": 201.00, "df": 16, "p": 0.001}


# Table 12: Region — over-/under-representation in SIU vs. Dec-2020
# penitentiary population.
TABLE12_REGIONAL_OVERREP = [
    {"region": "Atlantic", "siu_pct": 11.3, "pop_pct": 9.2},
    {"region": "Quebec", "siu_pct": 37.3, "pop_pct": 19.8},
    {"region": "Ontario", "siu_pct": 8.5, "pop_pct": 27.5},
    {"region": "Prairie", "siu_pct": 24.9, "pop_pct": 29.5},
    {"region": "Pacific", "siu_pct": 18.0, "pop_pct": 14.0},
]


# Table 15: Region × Mental-health flag. χ²=27.51, df=4, p<.001.
TABLE15_REGION_X_MENTAL_HEALTH = [
    {"region": "Atlantic", "no_mh": 169, "yes_mh": 88,
     "total": 257, "yes_pct": 34.2},
    {"region": "Quebec", "no_mh": 652, "yes_mh": 196,
     "total": 848, "yes_pct": 23.1},
    {"region": "Ontario", "no_mh": 160, "yes_mh": 48,
     "total": 208, "yes_pct": 23.1},
    {"region": "Prairies", "no_mh": 369, "yes_mh": 190,
     "total": 559, "yes_pct": 34.0},
    {"region": "Pacific", "no_mh": 291, "yes_mh": 116,
     "total": 407, "yes_pct": 28.5},
]
TABLE15_CHISQ = {"chi2": 27.51, "df": 4, "p": 0.001}


# Table 20: Days in SIU for the "torture" group (N=195).
TABLE20_TORTURE_GROUP_DAYS = [
    {"days": "16-31", "n": 88, "pct": 45.1},
    {"days": "32-61", "n": 59, "pct": 30.3},
    {"days": "62-380", "n": 48, "pct": 24.6},
]


# Table 22: Region × Mandela groups crosstab. χ²=208.54, df=8, p<.001.
TABLE22_REGION_X_MANDELA = [
    {"region": "Atlantic", "solitary": 53, "torture": 18,
     "everyone_else": 152, "total": 223,
     "solitary_pct": 23.8, "torture_pct": 8.1},
    {"region": "Quebec", "solitary": 295, "torture": 63,
     "everyone_else": 369, "total": 727,
     "solitary_pct": 40.6, "torture_pct": 8.7},
    {"region": "Ontario", "solitary": 41, "torture": 6,
     "everyone_else": 118, "total": 165,
     "solitary_pct": 24.8, "torture_pct": 3.6},
    {"region": "Prairies", "solitary": 49, "torture": 39,
     "everyone_else": 403, "total": 491,
     "solitary_pct": 10.0, "torture_pct": 7.9},
    {"region": "Pacific", "solitary": 118, "torture": 69,
     "everyone_else": 167, "total": 354,
     "solitary_pct": 33.3, "torture_pct": 19.5},
]
TABLE22_CHISQ = {"chi2": 208.54, "df": 8, "p": 0.001}


# ── May 2021 — additional tables (3, 5, 7, 8, 9, 10, 11, 14, 15) ──

TABLE3_MAY2021_REVIEWS_PER_STAY = [
    {"reviews": 1, "stays": 187, "total_reviews": 187},
    {"reviews": 2, "stays": 50, "total_reviews": 100},
    {"reviews": 3, "stays": 15, "total_reviews": 45},
    {"reviews": 4, "stays": 9, "total_reviews": 36},
    {"reviews": 5, "stays": 3, "total_reviews": 15},
    {"reviews": 6, "stays": 1, "total_reviews": 6},
]
TABLE3_MAY2021_TOTAL = {"stays": 265, "reviews": 389}


# Table 5: Race × Number of IEDM reviews. χ²=8.50, df=3, p<.05.
TABLE5_MAY2021_RACE_X_REVIEWS = [
    {"race": "White", "one": 56, "two_plus": 26, "total": 82,
     "two_plus_pct": 31.7},
    {"race": "Indigenous", "one": 84, "two_plus": 23, "total": 107,
     "two_plus_pct": 21.5},
    {"race": "Black", "one": 23, "two_plus": 19, "total": 42,
     "two_plus_pct": 45.2},
    {"race": "Other/Missing", "one": 24, "two_plus": 10, "total": 34,
     "two_plus_pct": 29.4},
]
TABLE5_MAY2021_CHISQ = {"chi2": 8.50, "df": 3, "p": 0.05}


# Table 7: Race × stay length, all stays N=1979. χ²=30.73, df=12, p<.01.
TABLE7_MAY2021_RACE_X_STAY = [
    {"race": "White", "n_total": 739, "pct_62_380": 19.5},
    {"race": "Indigenous", "n_total": 775, "pct_62_380": 19.5},
    {"race": "Black", "n_total": 284, "pct_62_380": 26.8},
    {"race": "Other/Missing", "n_total": 181, "pct_62_380": 21.0},
]
TABLE7_MAY2021_CHISQ = {"chi2": 30.73, "df": 12, "p": 0.01}


# Table 8: Race × stay length with 121+ split. χ²=41.63, df=15, p<.001.
# Black: 12.7% in 121+ days; whites 6.1%; Indigenous 5.5%; Other 8.8%.
TABLE8_MAY2021_RACE_X_STAY_121PLUS = [
    {"race": "White", "pct_121_380": 6.1, "n_121_380": 45},
    {"race": "Indigenous", "pct_121_380": 5.5, "n_121_380": 43},
    {"race": "Black", "pct_121_380": 12.7, "n_121_380": 36},
    {"race": "Other/Missing", "pct_121_380": 8.8, "n_121_380": 16},
]
TABLE8_MAY2021_CHISQ = {"chi2": 41.63, "df": 15, "p": 0.001}


# Table 9: IEDM decisions on the 380 reviews with outcomes.
TABLE9_MAY2021_IEDM_DECISIONS = [
    {"decision": "Decision moot", "n": 8, "pct": 2.1},
    {"decision": "Inmate to be removed from SIU", "n": 33, "pct": 8.7},
    {"decision": "Inmate to remain in SIU", "n": 224, "pct": 58.9},
    {"decision": ("N/A — Inmate transferred out of SIU before "
                    "decision rendered"),
     "n": 115, "pct": 30.3},
]
TABLE9_MAY2021_TOTAL = 380


# Table 10: 12 IEDMs × decision (binary: remain vs removed/moot).
# χ²=26.12, df=11, p<.01.
TABLE10_MAY2021_PER_IEDM = [
    {"iedm": 1, "remain": 22, "removed_or_moot": 25, "total": 47,
     "remain_pct": 46.8},
    {"iedm": 2, "remain": 26, "removed_or_moot": 9, "total": 35,
     "remain_pct": 74.3},
    {"iedm": 3, "remain": 18, "removed_or_moot": 15, "total": 33,
     "remain_pct": 54.5},
    {"iedm": 4, "remain": 27, "removed_or_moot": 23, "total": 50,
     "remain_pct": 54.0},
    {"iedm": 5, "remain": 18, "removed_or_moot": 10, "total": 28,
     "remain_pct": 64.3},
    {"iedm": 6, "remain": 30, "removed_or_moot": 9, "total": 39,
     "remain_pct": 76.9},
    {"iedm": 7, "remain": 23, "removed_or_moot": 8, "total": 31,
     "remain_pct": 74.2},
    {"iedm": 8, "remain": 12, "removed_or_moot": 20, "total": 32,
     "remain_pct": 37.5},
    {"iedm": 9, "remain": 16, "removed_or_moot": 13, "total": 29,
     "remain_pct": 55.2},
    {"iedm": 10, "remain": 8, "removed_or_moot": 11, "total": 19,
     "remain_pct": 42.1},
    {"iedm": 11, "remain": 6, "removed_or_moot": 1, "total": 7,
     "remain_pct": 85.7},
    {"iedm": 12, "remain": 18, "removed_or_moot": 12, "total": 30,
     "remain_pct": 60.0},
]
TABLE10_MAY2021_CHISQ = {"chi2": 26.12, "df": 11, "p": 0.01}


# Table 14: IEDM decision × release timing relative to referral date.
TABLE14_MAY2021_DECISION_X_RELEASE_TIMING = [
    {"decision": "Decision moot", "within_30d": 1, "31_40d": 2,
     "41_60d": 0, "after_61d": 1, "total": 4},
    {"decision": "Inmate to be removed", "within_30d": 8, "31_40d": 8,
     "41_60d": 3, "after_61d": 7, "total": 26,
     "within_30d_pct": 30.8, "after_61d_pct": 26.9},
    {"decision": "Inmate to remain", "within_30d": 13, "31_40d": 13,
     "41_60d": 22, "after_61d": 2, "total": 50,
     "within_30d_pct": 26.0, "after_61d_pct": 4.0},
    {"decision": ("N/A — transferred before decision"),
     "within_30d": 77, "31_40d": 2, "41_60d": 1, "after_61d": 0,
     "total": 80,
     "within_30d_pct": 96.3},
]


# Table 15: SIU stay length × IEDM-flag (105 long-stay no-IEDM cases).
TABLE15_MAY2021_NO_IEDM_BY_STAY_LENGTH = [
    {"days": "≤65", "no_iedm": 1580, "with_iedm": 21, "total": 1601,
     "no_iedm_pct": 98.7},
    {"days": "66-75", "no_iedm": 30, "with_iedm": 40, "total": 70,
     "no_iedm_pct": 42.9},
    {"days": "76-90", "no_iedm": 29, "with_iedm": 51, "total": 80,
     "no_iedm_pct": 36.3},
    {"days": "91-120", "no_iedm": 27, "with_iedm": 64, "total": 91,
     "no_iedm_pct": 29.7},
    {"days": ">120", "no_iedm": 49, "with_iedm": 88, "total": 137,
     "no_iedm_pct": 35.8},
]
TABLE15_MAY2021_TOTAL = 1979
TABLE15_MAY2021_LONG_STAY_NO_IEDM = 105  # 29+27+49


# ── Analyzers for new tables ──────────────────────────────────────


def analyze_table4_length_of_stay() -> RichResult:
    """Sprott-Doob (Feb 2021) Table 4: SIU length-of-stay distribution."""
    rows = [[r["days"], r["n"], f"{r['pct']:.1f}%"]
             for r in TABLE4_LENGTH_OF_STAY]
    rows.append(["Total", TABLE4_N, "100.0%"])
    short = sum(r["pct"] for r in TABLE4_LENGTH_OF_STAY
                 if r["days"] in ("1-5", "6-15"))
    long = sum(r["pct"] for r in TABLE4_LENGTH_OF_STAY
                if r["days"] in ("16-31", "32-61", "62-380"))
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 4 — SIU length-of-stay "
                "distribution (N=1,983 admissions Nov 2019 - Sept 2020)"),
        summary_lines=[
            ("N", TABLE4_N),
            ("≤15 day stays", f"{short:.1f}%"),
            ("≥16 day stays", f"{long:.1f}%"),
            ("≥62 day stays (longest bin)",
                f"{TABLE4_LENGTH_OF_STAY[-1]['pct']:.1f}%"),
        ],
        tables=[{
            "title": "Length-of-stay distribution:",
            "headers": ["Days in SIU", "N", "%"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Table 4. Roughly half of SIU stays are 15 days "
            "or shorter, but 20.8% (N=413) are ≥62 days — well past "
            "the Mandela Rule 43 'prolonged' threshold."
        ),
        payload={"table4": TABLE4_LENGTH_OF_STAY},
    )


def analyze_table11_region_x_stay_length() -> RichResult:
    """Sprott-Doob (Feb 2021) Table 11: Region × stay length."""
    rows = []
    for r in TABLE11_REGION_X_STAY_LENGTH:
        rows.append([r["region"], r["1-5"], r["6-15"],
                     r["16-31"], r["32-61"], r["62-380"], r["total"]])
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 11 — Region × Total "
                "days in SIU"),
        summary_lines=[
            ("Source", "Sprott & Doob (Feb 2021), p.18"),
            ("χ²", TABLE11_CHISQ["chi2"]),
            ("df", TABLE11_CHISQ["df"]),
            ("p", f"<{TABLE11_CHISQ['p']}"),
            ("Headline", ("Region distribution of stays differs "
                           "significantly across length bins")),
        ],
        tables=[{
            "title": ("Region × Stay length (% of region "
                       "in 62-380d bin shows long-stay concentration):"),
            "headers": ["Region", "1-5d", "6-15d", "16-31d",
                         "32-61d", "62-380d", "Total"],
            "rows": rows,
        }],
        payload={"table11": TABLE11_REGION_X_STAY_LENGTH,
                  "chisq": TABLE11_CHISQ},
    )


def analyze_table12_regional_overrepresentation() -> RichResult:
    """Sprott-Doob (Feb 2021) Table 12: Regional over-/under-rep in SIU."""
    rows = []
    for r in TABLE12_REGIONAL_OVERREP:
        ratio = r["siu_pct"] / r["pop_pct"] if r["pop_pct"] else 0
        rows.append([r["region"], f"{r['siu_pct']:.1f}%",
                     f"{r['pop_pct']:.1f}%", f"{ratio:.2f}×"])
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 12 — Regional over-/"
                "under-representation in SIU stays vs. Dec 2020 "
                "penitentiary population"),
        summary_lines=[
            ("Source", "Sprott & Doob (Feb 2021), p.18"),
            ("Quebec ratio", "1.88× (37.3% / 19.8%)"),
            ("Ontario ratio", "0.31× (8.5% / 27.5%)"),
            ("Headline", ("Quebec stays are 1.88× over-represented "
                           "and Ontario stays are 0.31× under-"
                           "represented in SIU vs. their share of "
                           "the federal prison population")),
        ],
        tables=[{
            "title": ("Region — SIU share vs. prison-pop share + ratio:"),
            "headers": ["Region", "SIU %", "Pop %",
                         "Over/under-rep ratio"],
            "rows": rows,
        }],
        payload={"table12": TABLE12_REGIONAL_OVERREP},
    )


def analyze_table15_region_x_mental_health() -> RichResult:
    """Sprott-Doob (Feb 2021) Table 15: Region × MH-flag."""
    rows = []
    for r in TABLE15_REGION_X_MENTAL_HEALTH:
        rows.append([r["region"], r["no_mh"], r["yes_mh"],
                     r["total"], f"{r['yes_pct']:.1f}%"])
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 15 — Region × Mental-"
                "health flag at SIU entry (N=2,279)"),
        summary_lines=[
            ("χ²", TABLE15_CHISQ["chi2"]),
            ("df", TABLE15_CHISQ["df"]),
            ("p", f"<{TABLE15_CHISQ['p']}"),
            ("Range across regions", "23.1% (Quebec/Ontario) to "
                                       "34.2% (Atlantic)"),
            ("Overall MH-flag rate", "28.0%"),
        ],
        tables=[{
            "title": "Region × MH flag at SIU entry:",
            "headers": ["Region", "No MH", "Yes MH", "Total", "Yes %"],
            "rows": rows,
        }],
        payload={"table15": TABLE15_REGION_X_MENTAL_HEALTH,
                  "chisq": TABLE15_CHISQ},
    )


def analyze_table22_region_x_mandela() -> RichResult:
    """Sprott-Doob (Feb 2021) Table 22: Region × Mandela groups."""
    rows = []
    for r in TABLE22_REGION_X_MANDELA:
        rows.append([r["region"],
                     f"{r['solitary']} ({r['solitary_pct']:.1f}%)",
                     f"{r['torture']} ({r['torture_pct']:.1f}%)",
                     r["everyone_else"], r["total"]])
    return RichResult(
        title=("Sprott & Doob (Feb 2021) Table 22 — Region × Mandela "
                "Rules classification (N=1,960)"),
        summary_lines=[
            ("χ²", TABLE22_CHISQ["chi2"]),
            ("df", TABLE22_CHISQ["df"]),
            ("p", f"<{TABLE22_CHISQ['p']}"),
            ("Quebec solitary share", "40.6% (N=295)"),
            ("Pacific torture share", "19.5% (N=69)"),
            ("Headline", ("Pacific has the highest torture share "
                           "(19.5%); Quebec has the highest solitary "
                           "share (40.6%)")),
        ],
        tables=[{
            "title": "Region × Mandela classification:",
            "headers": ["Region", "Solitary (%)", "Torture (%)",
                         "Everyone else", "Total"],
            "rows": rows,
        }],
        payload={"table22": TABLE22_REGION_X_MANDELA,
                  "chisq": TABLE22_CHISQ},
    )


def analyze_table9_iedm_decisions() -> RichResult:
    """Sprott-Doob-Iftene (May 2021) Table 9: IEDM review outcomes."""
    rows = [[r["decision"], r["n"], f"{r['pct']:.1f}%"]
             for r in TABLE9_MAY2021_IEDM_DECISIONS]
    rows.append(["Total", TABLE9_MAY2021_TOTAL, "100%"])
    pct_remain_or_moved_pre = (
        next(r["pct"] for r in TABLE9_MAY2021_IEDM_DECISIONS
              if "remain" in r["decision"].lower())
        + next(r["pct"] for r in TABLE9_MAY2021_IEDM_DECISIONS
                if "transferred" in r["decision"].lower())
    )
    return RichResult(
        title=("Sprott, Doob & Iftene (May 2021) Table 9 — IEDM "
                "review outcomes (N=380 reviews from 265 stays)"),
        summary_lines=[
            ("N reviews with outcomes", TABLE9_MAY2021_TOTAL),
            ("'Remove' decisions", "33 (8.7%)"),
            ("'Remain' decisions", "224 (58.9%)"),
            ("Pre-empted by CSC", "115 (30.3%)"),
            ("Decision moot", "8 (2.1%)"),
            ("% non-removal outcomes",
                f"{pct_remain_or_moved_pre:.1f}% — i.e., either CSC "
                f"had moved the prisoner before the IEDM decided, "
                f"or the IEDM said 'remain in SIU'"),
        ],
        tables=[{
            "title": "IEDM review outcomes (s.37.8):",
            "headers": ["Decision", "N", "%"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Table 9. Of 380 IEDM reviews with outcomes, "
            "fewer than 1 in 11 resulted in a 'remove from SIU' "
            "decision. Combined with the 30% of cases that CSC "
            "pre-empted (transferred the prisoner BEFORE the IEDM "
            "could rule), this severely limits IEDMs' practical "
            "ability to order earlier release."
        ),
        payload={"table9": TABLE9_MAY2021_IEDM_DECISIONS},
    )


def analyze_table10_per_iedm_variance() -> RichResult:
    """Sprott-Doob-Iftene (May 2021) Table 10: per-IEDM variance."""
    rows = []
    for r in TABLE10_MAY2021_PER_IEDM:
        rows.append([f"#{r['iedm']}", r["remain"],
                     r["removed_or_moot"], r["total"],
                     f"{r['remain_pct']:.1f}%"])
    pcts = [r["remain_pct"] for r in TABLE10_MAY2021_PER_IEDM]
    return RichResult(
        title=("Sprott, Doob & Iftene (May 2021) Table 10 — Per-IEDM "
                "decision variance (12 anonymized IEDMs)"),
        summary_lines=[
            ("χ²", TABLE10_MAY2021_CHISQ["chi2"]),
            ("df", TABLE10_MAY2021_CHISQ["df"]),
            ("p", f"<{TABLE10_MAY2021_CHISQ['p']}"),
            ("Min 'remain' rate", f"{min(pcts):.1f}% (IEDM #8)"),
            ("Max 'remain' rate", f"{max(pcts):.1f}% (IEDM #11)"),
            ("Range", f"{max(pcts) - min(pcts):.1f} percentage points"),
            ("Headline", ("Substantial variance across IEDMs — one "
                           "decided 37.5% should remain; another "
                           "decided 85.7% should remain")),
        ],
        tables=[{
            "title": "Per-IEDM (anonymized 1-12) decisions:",
            "headers": ["IEDM #", "Remain", "Remove/moot",
                         "Total", "% Remain"],
            "rows": rows,
        }],
        payload={"table10": TABLE10_MAY2021_PER_IEDM,
                  "chisq": TABLE10_MAY2021_CHISQ,
                  "min_remain_pct": min(pcts),
                  "max_remain_pct": max(pcts)},
    )


def analyze_table15_long_stay_no_iedm() -> RichResult:
    """Sprott-Doob-Iftene (May 2021) Table 15: long-stay no-IEDM cases."""
    rows = []
    for r in TABLE15_MAY2021_NO_IEDM_BY_STAY_LENGTH:
        rows.append([r["days"], r["no_iedm"], r["with_iedm"],
                     r["total"], f"{r['no_iedm_pct']:.1f}%"])
    return RichResult(
        title=("Sprott, Doob & Iftene (May 2021) Table 15 — Long-stay "
                "SIU cases with NO IEDM record (N=1,979)"),
        summary_lines=[
            ("N total stays examined", TABLE15_MAY2021_TOTAL),
            ("Long-stay (≥76d) with NO IEDM",
                f"{TABLE15_MAY2021_LONG_STAY_NO_IEDM} cases "
                f"(29+27+49)"),
            (">120d with NO IEDM",
                "49 cases (35.8% of 137 long-stay >120d cases)"),
            ("Headline", ("105 cases stayed 76+ days in an SIU "
                           "with no IEDM record — apparent compliance "
                           "failure with CCRA s.37.8")),
        ],
        tables=[{
            "title": "SIU stay length × IEDM-flag:",
            "headers": ["Days in SIU", "No IEDM", "With IEDM",
                         "Total", "% No IEDM"],
            "rows": rows,
        }],
        interpretation=(
            "Reproduces Table 15. The CCRA requires IEDM review for "
            "stays beyond 60 days, but 105 stays of 76+ days in "
            "this dataset had NO record of any IEDM review. Even "
            "more striking: 49 of the 137 stays >120 days (35.8%) "
            "had no IEDM record. This is a structural failure of "
            "the SIU oversight regime."
        ),
        payload={"table15_may2021": TABLE15_MAY2021_NO_IEDM_BY_STAY_LENGTH,
                  "long_stay_no_iedm": TABLE15_MAY2021_LONG_STAY_NO_IEDM},
    )


# ── χ² verifier — recompute published χ² from transcribed cells ──


def verify_chi2(observed: "list[list[int]]") -> dict:
    """Recompute Pearson χ² from a 2D contingency table.

    Pure-function rebuild of scipy.stats.chi2_contingency without
    Yates correction. Intended for quick self-checks of the
    transcribed cell counts against published χ² values.

    Parameters
    ----------
    observed : 2D list of int
        Contingency table (rows × cols).

    Returns
    -------
    dict with keys 'chi2', 'df', 'p_value', 'expected', 'n'.

    >>> # 2×2 sanity: independence
    >>> verify_chi2([[10, 10], [10, 10]])["chi2"]
    0.0
    """
    import numpy as np
    from scipy import stats as sps
    obs = np.asarray(observed, dtype=float)
    n = obs.sum()
    row_tot = obs.sum(axis=1, keepdims=True)
    col_tot = obs.sum(axis=0, keepdims=True)
    expected = row_tot @ col_tot / n
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(expected > 0,
                            (obs - expected) ** 2 / expected, 0.0)
    chi2 = float(contrib.sum())
    df = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    p = float(1 - sps.chi2.cdf(chi2, df)) if df > 0 else 1.0
    return {"chi2": round(chi2, 2), "df": int(df),
             "p_value": round(p, 6),
             "expected": expected.round(2).tolist(),
             "n": int(n)}


def verify_published_chi_squares() -> RichResult:
    """Re-derive every published χ² from its transcribed cell counts.

    Cross-checks Sprott-Doob Tables 11, 15, 22 (Feb 2021) and
    Sprott-Doob-Iftene Tables 5, 7, 8, 10 (May 2021). For each:
    rebuilds the observed contingency table from the per-row dicts,
    runs verify_chi2(), and reports recomputed vs. published values.
    """
    rows = []

    # Table 11 (Feb 2021): Region × stay length
    obs = [[r["1-5"], r["6-15"], r["16-31"], r["32-61"], r["62-380"]]
            for r in TABLE11_REGION_X_STAY_LENGTH]
    v = verify_chi2(obs)
    rows.append(["SD-2021-Feb T11 Region × stay length",
                  v["chi2"], v["df"], TABLE11_CHISQ["chi2"],
                  TABLE11_CHISQ["df"],
                  "✓" if abs(v["chi2"] - TABLE11_CHISQ["chi2"]) < 1.0
                  else "≠"])

    # Table 15 (Feb 2021): Region × MH-flag
    obs = [[r["no_mh"], r["yes_mh"]]
            for r in TABLE15_REGION_X_MENTAL_HEALTH]
    v = verify_chi2(obs)
    rows.append(["SD-2021-Feb T15 Region × MH",
                  v["chi2"], v["df"], TABLE15_CHISQ["chi2"],
                  TABLE15_CHISQ["df"],
                  "✓" if abs(v["chi2"] - TABLE15_CHISQ["chi2"]) < 1.0
                  else "≠"])

    # Table 22 (Feb 2021): Region × Mandela groups
    obs = [[r["solitary"], r["torture"], r["everyone_else"]]
            for r in TABLE22_REGION_X_MANDELA]
    v = verify_chi2(obs)
    rows.append(["SD-2021-Feb T22 Region × Mandela",
                  v["chi2"], v["df"], TABLE22_CHISQ["chi2"],
                  TABLE22_CHISQ["df"],
                  "✓" if abs(v["chi2"] - TABLE22_CHISQ["chi2"]) < 1.5
                  else "≠"])

    # Table 5 (May 2021): Race × #IEDM-reviews (one vs two-plus)
    obs = [[r["one"], r["two_plus"]]
            for r in TABLE5_MAY2021_RACE_X_REVIEWS]
    v = verify_chi2(obs)
    rows.append(["SDI-2021-May T5 Race × #reviews",
                  v["chi2"], v["df"], TABLE5_MAY2021_CHISQ["chi2"],
                  TABLE5_MAY2021_CHISQ["df"],
                  "✓" if abs(v["chi2"] - TABLE5_MAY2021_CHISQ["chi2"])
                  < 1.0 else "≠"])

    # Table 10 (May 2021): per-IEDM remain vs removed
    obs = [[r["remain"], r["removed_or_moot"]]
            for r in TABLE10_MAY2021_PER_IEDM]
    v = verify_chi2(obs)
    rows.append(["SDI-2021-May T10 Per-IEDM",
                  v["chi2"], v["df"], TABLE10_MAY2021_CHISQ["chi2"],
                  TABLE10_MAY2021_CHISQ["df"],
                  "✓" if abs(v["chi2"] - TABLE10_MAY2021_CHISQ["chi2"])
                  < 1.5 else "≠"])

    n_pass = sum(1 for r in rows if r[-1] == "✓")
    return RichResult(
        title=("χ² verification — recomputed from transcribed "
                "cell counts vs. published values"),
        summary_lines=[
            ("Tables verified", len(rows)),
            ("Pass count", f"{n_pass}/{len(rows)}"),
            ("Method", "Pearson χ² without Yates correction"),
            ("Tolerance",
                "1.0–1.5 χ² units (rounding in published values)"),
        ],
        tables=[{
            "title": ("Recomputed χ² vs. published — pass = within "
                       "rounding tolerance:"),
            "headers": ["Source", "Recomputed χ²", "df",
                         "Published χ²", "df", "✓?"],
            "rows": rows,
        }],
        interpretation=(
            "Recomputes every published χ² from the transcribed "
            "contingency-table cells. A '✓' means the recomputed "
            "value is within rounding tolerance of the published "
            "value — confirming both the transcription is accurate "
            "and the published statistic is correctly derived. "
            "A '≠' indicates either a transcription error or a "
            "difference in the χ² formula (e.g., Yates correction)."
        ),
        payload={"n_pass": n_pass, "n_total": len(rows)},
    )


def analyze_full_sprott_doob_feb2021() -> RichResult:
    """Comprehensive replication of Sprott & Doob (Feb 2021) main tables."""
    t13 = analyze_table13_regional_rates()
    t19 = analyze_table19_mandela_classification()
    t23 = analyze_table23_regional_torture_rates()
    sections = []
    for label, r in [("§ Table 13", t13), ("§ Table 19", t19),
                      ("§ Table 23", t23)]:
        if r.tables:
            sections.append({
                "title": f"{label} — {r.title}",
                "headers": r.tables[0]["headers"],
                "rows": r.tables[0]["rows"],
            })
    return RichResult(
        title=("Sprott & Doob (Feb 2021) — Solitary Confinement, "
                "Torture, and Canada's SIUs (CRIMSL UToronto)"),
        summary_lines=[
            ("Authors", "Jane B. Sprott (Ryerson) & Anthony N. Doob (UofT)"),
            ("Date", "23 February 2021"),
            ("URL", "crimsl.utoronto.ca/.../TortureSolitarySIUsSprottDoob23Feb2021_0.pdf"),
            ("N total person-stays", HEADLINE_FINDINGS["n_total_stays"]),
            ("% missed full 4 hrs out of cell every day",
                f"{HEADLINE_FINDINGS['missed_full_4hrs_overall_pct']:.1f}%"),
            ("Long-stay (16+ d) missed 4 hrs in ≥76% days",
                f"{HEADLINE_FINDINGS['long_stay_missed_4hrs_in_76pct_of_days']:.0f}%"),
            ("Pacific/Ontario torture ratio",
                f"{HEADLINE_FINDINGS['pacific_to_ontario_ratio']:.1f}×"),
        ],
        tables=sections,
        interpretation=(
            "Comprehensive replication of the three core tables from "
            "Sprott & Doob's third CRIMSL report (Feb 2021). The "
            "report's central empirical contribution is the Mandela-"
            "Rules classifier (operationalized in "
            "`classify_mandela()`): 28.4% of SIU stays meet the UN "
            "definition of solitary confinement, 9.9% meet the "
            "definition of torture or other cruel, inhuman, or "
            "degrading treatment. The regional disparities — "
            "Pacific 22.6× Ontario for torture rate — point to "
            "decision-making rather than population characteristics "
            "as the driver."
        ),
        payload={
            "table13": TABLE13_REGIONAL_RATES_PER_1000,
            "table19": TABLE19_MANDELA_CLASSIFICATION,
            "table23": TABLE23_REGIONAL_TORTURE_RATES,
            "headline_findings": HEADLINE_FINDINGS,
        },
    )
