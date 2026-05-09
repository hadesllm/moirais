"""
Canonical tests for moirais.envhealth — the pollution-health composition
module. Each function has at least one test against a published
invariant (citation in the docstring) plus basic structural checks.

Philosophy mirrors test_causal_canonical.py: a failing test here is a
real math bug, not a template regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# concentration_response_pm25 — Burnett et al. (2014) IER
# ---------------------------------------------------------------------------

class TestCRFPm25:
    """Reference: Burnett et al. (2014) EHP 122(4):397-403, Eq. 1."""

    def test_rr_is_one_at_counterfactual(self) -> None:
        """At z = z_cf, excess = 0 → RR must equal 1 exactly."""
        from moirais.envhealth import concentration_response_pm25
        r = concentration_response_pm25(exposure=5.8, reference_conc=5.8)
        assert r.rr == pytest.approx(1.0, abs=1e-10)
        assert r.log_rr == pytest.approx(0.0, abs=1e-10)

    def test_rr_is_one_below_counterfactual(self) -> None:
        """Below z_cf, no excess risk: RR clips to 1 (IER convention)."""
        from moirais.envhealth import concentration_response_pm25
        r = concentration_response_pm25(exposure=3.0, reference_conc=5.8)
        assert r.rr == pytest.approx(1.0, abs=1e-10)

    def test_rr_monotone_in_exposure(self) -> None:
        from moirais.envhealth import concentration_response_pm25
        rr_vals = [
            concentration_response_pm25(z, reference_conc=5.8).rr
            for z in (8, 12, 20, 35, 60)
        ]
        for a, b in zip(rr_vals, rr_vals[1:]):
            assert b >= a - 1e-10, (
                f"RR non-monotone in PM2.5: {a} -> {b}")

    def test_rr_structure_at_moderate_exposure(self) -> None:
        """RR at 20 μg/m³ should be > 1 and < α+1 (asymptote)."""
        from moirais.envhealth import concentration_response_pm25
        r = concentration_response_pm25(exposure=20.0, reference_conc=5.8)
        alpha = r.extra["alpha"]
        assert 1.0 < r.rr < 1.0 + alpha, (
            f"RR at PM2.5=20 μg/m³ = {r.rr}, expected in (1, {1+alpha}).")

    def test_unknown_outcome_raises(self) -> None:
        from moirais.envhealth import concentration_response_pm25
        with pytest.raises(ValueError, match="outcome"):
            concentration_response_pm25(10.0, outcome="not_an_outcome")


# ---------------------------------------------------------------------------
# concentration_response_no2 — log-linear
# ---------------------------------------------------------------------------

class TestCRFNo2:
    """Reference: Atkinson et al. (2018); WHO (2021)."""

    def test_rr_is_one_at_counterfactual(self) -> None:
        from moirais.envhealth import concentration_response_no2
        r = concentration_response_no2(exposure=10.0, reference_conc=10.0)
        assert r.rr == pytest.approx(1.0, abs=1e-10)
        assert r.log_rr == pytest.approx(0.0, abs=1e-10)

    def test_doubling_increment_doubles_log_rr(self) -> None:
        """Log-linear: log-RR scales linearly with (z - z_cf)."""
        from moirais.envhealth import concentration_response_no2
        r1 = concentration_response_no2(exposure=20.0, reference_conc=10.0)
        r2 = concentration_response_no2(exposure=30.0, reference_conc=10.0)
        assert r2.log_rr == pytest.approx(2 * r1.log_rr, rel=1e-10)

    def test_beta_override_applied(self) -> None:
        """Explicit beta_per_10 overrides the outcome-keyed lookup."""
        from moirais.envhealth import concentration_response_no2
        r = concentration_response_no2(exposure=20.0, reference_conc=10.0,
                                       beta_per_10=0.1)
        # log-RR = 0.1 * (20-10)/10 = 0.1
        assert r.log_rr == pytest.approx(0.1, rel=1e-10)
        assert r.rr == pytest.approx(np.exp(0.1), rel=1e-10)


# ---------------------------------------------------------------------------
# attributable_fraction — Levin (1953) / Rothman et al. (2008)
# ---------------------------------------------------------------------------

class TestAttributableFraction:
    """Reference: Rothman, Greenland & Lash (2008) Ch 5."""

    def test_paf_zero_when_prevalence_zero(self) -> None:
        from moirais.envhealth import attributable_fraction
        assert attributable_fraction(rr=2.0, exposure_prevalence=0.0) == 0.0

    def test_paf_zero_when_rr_is_one(self) -> None:
        from moirais.envhealth import attributable_fraction
        assert attributable_fraction(rr=1.0, exposure_prevalence=0.5) == 0.0

    def test_paf_identity_at_p_equals_one(self) -> None:
        """At p=1, PAF = (RR-1)/RR (everyone exposed; preventable
        fraction is the risk difference / risk)."""
        from moirais.envhealth import attributable_fraction
        for rr in (1.5, 2.0, 3.5):
            got = attributable_fraction(rr=rr, exposure_prevalence=1.0)
            expected = (rr - 1.0) / rr
            assert got == pytest.approx(expected, rel=1e-10)

    def test_paf_monotone_in_prevalence(self) -> None:
        """For harmful exposure (RR > 1), PAF is monotone increasing in p."""
        from moirais.envhealth import attributable_fraction
        pafs = [attributable_fraction(2.0, p) for p in (0.0, 0.2, 0.5, 0.8, 1.0)]
        for a, b in zip(pafs, pafs[1:]):
            assert b >= a - 1e-12

    def test_paf_rejects_out_of_range_prevalence(self) -> None:
        from moirais.envhealth import attributable_fraction
        with pytest.raises(ValueError):
            attributable_fraction(rr=2.0, exposure_prevalence=1.5)


# ---------------------------------------------------------------------------
# mortality_displaced — BenMAP-style
# ---------------------------------------------------------------------------

class TestMortalityDisplaced:
    """Reference: US EPA BenMAP-CE (2018); Anenberg et al. (2010)."""

    def test_zero_delta_gives_zero_displaced(self) -> None:
        from moirais.envhealth import mortality_displaced
        got = mortality_displaced(
            exposure_delta=0.0, population=1_000_000,
            baseline_rate=0.008, beta_per_unit=0.004)
        assert got == pytest.approx(0.0, abs=1e-12)

    def test_linear_scaling_with_population(self) -> None:
        """Deaths scale linearly with population (for fixed Δ, β, y₀)."""
        from moirais.envhealth import mortality_displaced
        kw = dict(exposure_delta=5.0, baseline_rate=0.008, beta_per_unit=0.004)
        deaths_1m = mortality_displaced(population=1_000_000, **kw)
        deaths_2m = mortality_displaced(population=2_000_000, **kw)
        assert deaths_2m == pytest.approx(2 * deaths_1m, rel=1e-10)

    def test_small_beta_approximation(self) -> None:
        """For small β·Δ, ΔY ≈ y₀·N·β·Δ (first-order Taylor)."""
        from moirais.envhealth import mortality_displaced
        y0, N, beta, delta = 0.008, 1_000_000, 0.001, 2.0
        exact = mortality_displaced(
            exposure_delta=delta, population=N,
            baseline_rate=y0, beta_per_unit=beta)
        approx = y0 * N * beta * delta
        # At β·Δ = 0.002, error should be < 0.5%
        assert abs(exact - approx) / approx < 0.005

    def test_rejects_negative_population(self) -> None:
        from moirais.envhealth import mortality_displaced
        with pytest.raises(ValueError, match="population"):
            mortality_displaced(
                exposure_delta=5.0, population=-100,
                baseline_rate=0.008, beta_per_unit=0.004)


# ---------------------------------------------------------------------------
# burden_of_pollution — end-to-end
# ---------------------------------------------------------------------------

class TestBurdenOfPollution:
    """Composition: CRF → PAF → attributable cases."""

    def test_pipeline_runs_end_to_end_no2(self) -> None:
        from moirais.envhealth import burden_of_pollution
        r = burden_of_pollution(
            exposure_mean=20.0, exposure_prevalence=1.0,
            baseline_rate=0.008, population=500_000,
            pollutant="NO2", outcome="all_cause_mortality",
        )
        assert r.paf > 0
        assert r.attributable_cases > 0
        assert r.baseline_cases == pytest.approx(0.008 * 500_000)
        # Attributable should be a fraction of baseline, never exceed it
        assert r.attributable_cases < r.baseline_cases

    def test_pipeline_pm25(self) -> None:
        from moirais.envhealth import burden_of_pollution
        r = burden_of_pollution(
            exposure_mean=15.0, exposure_prevalence=1.0,
            baseline_rate=0.008, population=500_000,
            pollutant="PM2.5", outcome="all_cause_mortality",
        )
        assert r.paf > 0
        assert r.pollutant == "PM2.5"

    def test_zero_exposure_above_reference_gives_zero_burden(self) -> None:
        """If exposure_mean == reference, RR=1 and PAF=0 → no burden."""
        from moirais.envhealth import burden_of_pollution
        r = burden_of_pollution(
            exposure_mean=10.0, exposure_prevalence=1.0,
            baseline_rate=0.008, population=500_000,
            pollutant="NO2", reference_conc=10.0,
        )
        assert r.paf == pytest.approx(0.0, abs=1e-10)
        assert r.attributable_cases == pytest.approx(0.0, abs=1e-10)

    def test_unknown_pollutant_raises(self) -> None:
        from moirais.envhealth import burden_of_pollution
        with pytest.raises(ValueError, match="pollutant"):
            burden_of_pollution(
                exposure_mean=20.0, exposure_prevalence=1.0,
                baseline_rate=0.008, population=1000, pollutant="FOO")


# ---------------------------------------------------------------------------
# exposure_response_sensitivity — DML + bootstrap
# ---------------------------------------------------------------------------

class TestExposureResponseSensitivity:
    """
    Canonical expectation: the analytic DML estimate and the bootstrap
    mean should agree within combined SE. Bootstrap CI should contain
    the true effect on a known-truth DGP.
    """

    def test_wraps_dml_with_bootstrap(self) -> None:
        from moirais.envhealth import exposure_response_sensitivity
        rng = np.random.default_rng(771)
        n, p = 600, 3
        tau = 1.0
        X = rng.normal(size=(n, p))
        T = rng.binomial(1, 0.5, size=n)
        Y = tau * T + X @ np.array([0.5, -0.3, 0.2]) + rng.normal(size=n)
        df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p)])
        df["T"] = T.astype(float)
        df["Y"] = Y

        result = exposure_response_sensitivity(
            df,
            outcome="Y", exposure="T",
            confounders=[f"X{i+1}" for i in range(p)],
            n_bootstrap=20,
        )
        assert {"ate", "se_analytic", "se_bootstrap",
                "ci_lower_bs", "ci_upper_bs", "n_bootstrap"}.issubset(result.keys())
        # Bootstrap CI should contain truth at 95% (single seed; loose)
        assert result["ci_lower_bs"] <= tau + 0.5
        assert result["ci_upper_bs"] >= tau - 0.5
        assert result["n_bootstrap"] >= 10


# ---------------------------------------------------------------------------
# pollution_equity_analysis — Wagstaff et al. (1991)
# ---------------------------------------------------------------------------

class TestPollutionEquityAnalysis:
    """Reference: Wagstaff, Paci & van Doorslaer (1991); O'Donnell et al. (2008)."""

    def test_ci_is_zero_for_uniform_exposure(self) -> None:
        """If exposure is identical across units, CI = 0 exactly."""
        from moirais.envhealth import pollution_equity_analysis
        rng = np.random.default_rng(801)
        df = pd.DataFrame({
            "exposure": np.full(500, 20.0),
            "income": rng.uniform(20000, 100000, size=500),
        })
        r = pollution_equity_analysis(df, exposure="exposure", income="income")
        assert r.concentration_index == pytest.approx(0.0, abs=1e-10)

    def test_ci_is_negative_when_low_income_worse_off(self) -> None:
        """Construct a DGP where exposure decreases with income. CI should
        be clearly negative (pro-poor exposure burden)."""
        from moirais.envhealth import pollution_equity_analysis
        rng = np.random.default_rng(802)
        income = rng.uniform(20000, 100000, size=2000)
        # Higher income → lower exposure (classic environmental-justice pattern)
        exposure = 30.0 - 0.00015 * income + rng.normal(scale=1.0, size=2000)
        df = pd.DataFrame({"exposure": exposure, "income": income})
        r = pollution_equity_analysis(df, exposure="exposure", income="income")
        assert r.concentration_index < -0.05, (
            f"CI = {r.concentration_index}, expected < -0.05 for pro-poor burden.")
        assert "pro-poor" in r.interpretation.lower()

    def test_ci_is_positive_when_high_income_worse_off(self) -> None:
        """Mirror test: exposure increases with income → CI > 0 (pro-rich)."""
        from moirais.envhealth import pollution_equity_analysis
        rng = np.random.default_rng(803)
        income = rng.uniform(20000, 100000, size=2000)
        exposure = 10.0 + 0.0001 * income + rng.normal(scale=1.0, size=2000)
        df = pd.DataFrame({"exposure": exposure, "income": income})
        r = pollution_equity_analysis(df, exposure="exposure", income="income")
        assert r.concentration_index > 0.05
        assert "pro-rich" in r.interpretation.lower()


# ---------------------------------------------------------------------------
# burden_by_fsa — stratified pipeline
# ---------------------------------------------------------------------------

class TestBurdenByFsa:
    """Apply burden_of_pollution per-row to an FSA table."""

    def test_returns_dataframe_sorted_by_attributable(self) -> None:
        from moirais.envhealth import burden_by_fsa
        fsa = pd.DataFrame([
            {"fsa": "M6H", "exposure": 22.0, "population": 35000, "baseline_rate": 0.008},
            {"fsa": "M5V", "exposure": 18.0, "population": 40000, "baseline_rate": 0.008},
            {"fsa": "M1B", "exposure": 14.0, "population": 60000, "baseline_rate": 0.008},
        ])
        result = burden_by_fsa(fsa, pollutant="NO2")
        # 3 rows, columns include paf + attributable_cases
        assert len(result) == 3
        assert {"fsa", "rr", "paf", "attributable_cases",
                "baseline_cases"}.issubset(result.columns)
        # Sorted descending by attributable_cases
        attrs = result["attributable_cases"].tolist()
        for a, b in zip(attrs, attrs[1:]):
            assert a >= b

    def test_missing_column_raises(self) -> None:
        from moirais.envhealth import burden_by_fsa
        bad = pd.DataFrame([{"fsa": "M6H", "exposure": 20.0}])
        with pytest.raises(ValueError, match="missing"):
            burden_by_fsa(bad)
