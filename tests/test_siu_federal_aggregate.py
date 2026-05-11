"""Tests for the federal-aggregate companion modules:
- morie.siuiap (SIU IAP citations + CRIMSL_REPORTS + AFFIDAVITS)
- morie.sprott_doob (CRIMSL Feb 2021 + Schulich May 2021 tables +
  Mandela classifier + χ² verifier)
- morie.doob_trends (CCRSO 2018 Tables 1-3 + Pettitt + decoupling)
"""

from __future__ import annotations

import pytest


# ── morie.siuiap ──────────────────────────────────────────────


class TestSiuIap:
    def test_panel_members_three(self):
        from morie import siuiap
        assert len(siuiap.PANEL_MEMBERS) == 3
        assert siuiap.PANEL_MEMBERS[0]["name"] == "Howard Sapers"
        assert siuiap.PANEL_MEMBERS[1]["name"] == "Anthony N. Doob"
        assert siuiap.PANEL_MEMBERS[2]["name"] == "Jane B. Sprott"

    def test_original_panel_2019_2020(self):
        from morie import siuiap
        p = siuiap.ORIGINAL_PANEL_2019_2020
        assert p["chair"] == "Anthony N. Doob"
        assert p["established"] == "2019"
        assert p["dissolved"] == "mid-2020"

    def test_reports_dict_has_5_panel_reports(self):
        from morie import siuiap
        assert len(siuiap.REPORTS) == 5
        for r in siuiap.REPORTS.values():
            assert "title" in r
            assert "year" in r

    def test_crimsl_reports_dict_has_4_papers(self):
        from morie import siuiap
        assert len(siuiap.CRIMSL_REPORTS) == 4
        for r in siuiap.CRIMSL_REPORTS.values():
            assert "Sprott" in " ".join(r["Knowing yourself is the beginning of all wisdom. — Aristotle"])

    def test_affidavits_has_doob(self):
        from morie import siuiap
        assert "doob_t_539_20_2020" in siuiap.AFFIDAVITS

    def test_cite_panel_report(self):
        from morie import siuiap
        s = siuiap.cite("final_2024")
        assert "SIU IAP" in s
        assert "2024" in s

    def test_cite_crimsl_report(self):
        from morie import siuiap
        s = siuiap.cite("sprott_doob_torture_solitary_2021")
        assert "Sprott" in s
        assert "Doob" in s
        assert "2021" in s

    def test_cite_iftene_paper(self):
        from morie import siuiap
        s = siuiap.cite("sprott_doob_iftene_external_decision_makers_2021")
        assert "Iftene" in s

    def test_cite_unknown_raises(self):
        from morie import siuiap
        with pytest.raises(KeyError):
            siuiap.cite("nonexistent_report_id")


# ── morie.sprott_doob — Mandela classifier ────────────────────


class TestMandelaClassifier:
    def test_torture(self):
        from morie.sprott_doob import classify_mandela
        r = classify_mandela(20, 1.5, 100)
        assert r["category"] == "Torture"
        assert "Mandela Rules 43+44" in r["rule"]

    def test_solitary(self):
        from morie.sprott_doob import classify_mandela
        r = classify_mandela(8, 1.5, 100)
        assert r["category"] == "Solitary Confinement"
        assert "Rule 44" in r["rule"]

    def test_all_other_too_many_hours(self):
        from morie.sprott_doob import classify_mandela
        r = classify_mandela(20, 5, 100)
        assert r["category"] == "All other"

    def test_all_other_partial_missed(self):
        from morie.sprott_doob import classify_mandela
        r = classify_mandela(20, 1.5, 50)
        assert r["category"] == "All other"

    def test_boundary_15_days(self):
        from morie.sprott_doob import classify_mandela
        r = classify_mandela(15, 1.5, 100)
        assert r["category"] == "Solitary Confinement"

    def test_boundary_16_days(self):
        from morie.sprott_doob import classify_mandela
        r = classify_mandela(16, 1.5, 100)
        assert r["category"] == "Torture"


# ── morie.sprott_doob — χ² verifier ───────────────────────────


class TestChiSquareVerifier:
    def test_independence_2x2(self):
        from morie.sprott_doob import verify_chi2
        r = verify_chi2([[10, 10], [10, 10]])
        assert r["chi2"] == 0.0
        assert r["df"] == 1

    def test_perfect_dependence(self):
        from morie.sprott_doob import verify_chi2
        r = verify_chi2([[100, 0], [0, 100]])
        assert r["chi2"] == 200.0

    def test_published_chi_squares_all_pass(self):
        from morie.sprott_doob import verify_published_chi_squares
        r = verify_published_chi_squares()
        assert r.payload["n_pass"] == r.payload["n_total"]
        assert r.payload["n_total"] >= 5

    def test_chi_square_rebuilds_to_published(self):
        """Sanity: cell counts in TABLE11 reproduce χ²=201.00."""
        from morie.sprott_doob import (
            TABLE11_REGION_X_STAY_LENGTH, verify_chi2,
        )
        obs = [[r["1-5"], r["6-15"], r["16-31"],
                 r["32-61"], r["62-380"]]
                for r in TABLE11_REGION_X_STAY_LENGTH]
        v = verify_chi2(obs)
        assert abs(v["chi2"] - 201.0) < 1.0
        assert v["df"] == 16


# ── morie.sprott_doob — analyzers ─────────────────────────────


class TestSprottDoobAnalyzers:
    @pytest.mark.parametrize("fn_name", [
        "analyze_table4_length_of_stay",
        "analyze_table11_region_x_stay_length",
        "analyze_table12_regional_overrepresentation",
        "analyze_table15_region_x_mental_health",
        "analyze_table22_region_x_mandela",
        "analyze_table9_iedm_decisions",
        "analyze_table10_per_iedm_variance",
        "analyze_table15_long_stay_no_iedm",
        "analyze_iedm_table1_population",
        "analyze_iedm_review_outcomes",
        "analyze_table13_regional_rates",
        "analyze_table19_mandela_classification",
        "analyze_table23_regional_torture_rates",
        "analyze_full_sprott_doob_feb2021",
    ])
    def test_analyzer_returns_richresult_with_content(self, fn_name):
        from morie import sprott_doob
        fn = getattr(sprott_doob, fn_name)
        r = fn()
        assert r.title  # non-empty
        # Must have either summary lines OR tables (not both empty)
        has_content = bool(r.summary_lines) or bool(r.tables)
        assert has_content, f"{fn_name} returned empty RichResult"


class TestSprottDoobTableConstants:
    def test_table19_sums_to_n_1960(self):
        from morie.sprott_doob import TABLE19_MANDELA_CLASSIFICATION
        total = sum(r["n"] for r in TABLE19_MANDELA_CLASSIFICATION)
        assert total == 1960

    def test_table19_percentages_sum_to_100(self):
        from morie.sprott_doob import TABLE19_MANDELA_CLASSIFICATION
        total_pct = sum(r["percent"]
                         for r in TABLE19_MANDELA_CLASSIFICATION)
        assert abs(total_pct - 100.0) < 0.5

    def test_table1_iedm_population_n_265(self):
        from morie.sprott_doob import TABLE1_IEDM_POPULATION
        for category in ("gender", "age_group", "race", "mental_health"):
            n_sum = sum(r["n"] for r in TABLE1_IEDM_POPULATION[category])
            assert n_sum == 265, (
                f"{category}: expected 265, got {n_sum}")

    def test_table10_iedms_count_12(self):
        from morie.sprott_doob import TABLE10_MAY2021_PER_IEDM
        assert len(TABLE10_MAY2021_PER_IEDM) == 12

    def test_pacific_torture_22_6x_ontario(self):
        from morie.sprott_doob import TABLE23_REGIONAL_TORTURE_RATES
        pac = next(r["torture_rate"]
                    for r in TABLE23_REGIONAL_TORTURE_RATES
                    if r["region"] == "Pacific")
        ont = next(r["torture_rate"]
                    for r in TABLE23_REGIONAL_TORTURE_RATES
                    if r["region"] == "Ontario")
        ratio = pac / ont
        assert 22.0 < ratio < 23.0


# ── morie.doob_trends ─────────────────────────────────────────


class TestDoobTrends:
    def test_table1_releases_5_year_avg_consistent(self):
        from morie.doob_trends import CCRSO_TABLE1_RELEASES
        # Verify violent revocation rate < 1% overall
        total = sum(r["total"] for r in CCRSO_TABLE1_RELEASES)
        violent = sum(r["revoke_violent"]
                       for r in CCRSO_TABLE1_RELEASES)
        rate_pct = 100 * violent / total
        assert rate_pct < 1.0

    def test_table2_flow_5_years(self):
        from morie.doob_trends import CCRSO_TABLE2_FLOW
        assert len(CCRSO_TABLE2_FLOW) == 5

    def test_table3_age_canada_total_29M(self):
        from morie.doob_trends import CCRSO_TABLE3_AGE
        total = sum(r["canada_adult_pop"] for r in CCRSO_TABLE3_AGE)
        assert 29_000_000 < total < 30_000_000

    def test_pettitt_changepoint_detects_step(self):
        from morie.doob_trends import pettitt_changepoint
        # Use n=20 so the asymptotic p-value approximation is valid.
        series = [1] * 10 + [5] * 10
        r = pettitt_changepoint(series)
        # change-point at index 9 (last index of regime 1)
        assert r["change_point_index"] in (8, 9, 10)
        assert r["p_value"] < 0.05

    def test_pettitt_short_series_returns_nan(self):
        from morie.doob_trends import pettitt_changepoint
        r = pettitt_changepoint([1, 2, 3])
        assert r["change_point_index"] is None

    def test_decoupling_test_uncorrelated_yields_small_r(self):
        import numpy as np
        from morie.doob_trends import decoupling_test
        rng = np.random.default_rng(42)
        crime = list(rng.normal(8000, 100, 30))
        imp = list(rng.normal(120, 5, 30))
        r = decoupling_test(crime, imp, years=range(1990, 2020))
        assert abs(r.payload["r_pearson"]) < 0.5

    def test_decoupling_length_mismatch_warns(self):
        from morie.doob_trends import decoupling_test
        r = decoupling_test([1, 2, 3], [1, 2, 3, 4], years=None)
        assert r.warnings

    def test_doob_full_affidavit_has_3_tables(self):
        from morie.doob_trends import analyze_doob_full_affidavit
        r = analyze_doob_full_affidavit()
        assert len(r.tables) == 3


# ── master report integration ───────────────────────────────────


class TestMasterReportIntegration:
    def test_master_has_at_least_4_sections(self):
        from morie.otis_all_analyze import analyze_ruhela_master
        r = analyze_ruhela_master(include_per_row=False)
        assert len(r.tables) >= 4

    def test_master_includes_sprott_doob_section(self):
        from morie.otis_all_analyze import analyze_ruhela_master
        r = analyze_ruhela_master(include_per_row=False)
        titles = " ".join(t["title"] for t in r.tables)
        assert "Sprott-Doob" in titles or "Sprott" in titles

    def test_master_includes_doob_t539_section(self):
        from morie.otis_all_analyze import analyze_ruhela_master
        r = analyze_ruhela_master(include_per_row=False)
        titles = " ".join(t["title"] for t in r.tables)
        assert "T-539-20" in titles or "Doob" in titles

    def test_master_includes_mandela_rf_section(self):
        from morie.otis_all_analyze import analyze_ruhela_master
        r = analyze_ruhela_master(include_per_row=False)
        titles = " ".join(t["title"] for t in r.tables)
        assert "Mandela-RF" in titles


# ── Mandela-RF on OTIS provincial data ────────────────────────────


class TestMandelaRF:
    def test_b05_per_placement_returns_yearly_rows(self):
        from morie.otis_all_analyze import (
            analyze_b05_mandela_classification,
        )
        r = analyze_b05_mandela_classification()
        assert r.tables
        assert len(r.tables[0]["rows"]) >= 1
        for row in r.tables[0]["rows"]:
            # Each row: [year, sol_n, sol_pct, tor_n, tor_pct, total]
            assert isinstance(row[0], int)
            assert row[1] + row[3] == row[5]  # solitary + torture = total

    def test_c11_per_individual_returns_two_views(self):
        from morie.otis_all_analyze import (
            analyze_c11_mandela_classification,
        )
        r = analyze_c11_mandela_classification()
        types = {row[1] for row in r.tables[0]["rows"]}
        assert "Segregation" in types
        assert "Restrictive confinement" in types

    def test_provincial_vs_federal_gap_computed(self):
        from morie.otis_all_analyze import (
            analyze_otis_mandela_provincial_vs_federal,
        )
        r = analyze_otis_mandela_provincial_vs_federal()
        assert "gap_pp" in r.payload
        assert r.payload["federal"]["torture_pct"] == 9.9
        assert r.payload["federal"]["solitary_pct"] == 28.4
        assert r.payload["federal"]["n"] == 1960

    def test_b05_torture_proportion_under_5_percent(self):
        """OTIS b05 (per-placement) should show very low torture
        proportion because most placements are short."""
        from morie.otis_all_analyze import (
            analyze_b05_mandela_classification,
        )
        r = analyze_b05_mandela_classification()
        # Each row has torture % at index 4
        for row in r.tables[0]["rows"]:
            tor_pct = float(row[4].rstrip("%"))
            # b05 unit is placement, dominated by 1-day; torture % low
            assert tor_pct < 5.0

    def test_c11_segregation_torture_at_least_5_percent(self):
        """OTIS c11 segregation view should show non-trivial
        Mandela-classified torture rate (individuals integrate
        across multiple placements)."""
        from morie.otis_all_analyze import (
            analyze_c11_mandela_classification,
        )
        r = analyze_c11_mandela_classification()
        seg_rows = [row for row in r.tables[0]["rows"]
                     if row[1] == "Segregation"]
        assert seg_rows
        for row in seg_rows:
            tor_pct = float(row[5].rstrip("%"))
            assert tor_pct >= 5.0  # always > 5% across years
