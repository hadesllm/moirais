"""
Comprehensive tests for six MORIE causal inference modules.

Modules tested:
    1. morie.did        — Difference-in-Differences
    2. morie.iv         — Instrumental Variables
    3. morie.rdd        — Regression Discontinuity Design
    4. morie.matching   — Matching estimators
    5. morie.sensitivity — Sensitivity analysis
    6. morie.survival   — Survival analysis

All tests use synthetic data generated with np.random.default_rng(42).
No external files or CPADS data are required.

References
----------
Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*.
Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics*.
Collett, D. (2015). *Modelling Survival Data in Medical Research* (3rd ed.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from morie.did import (
    did_2x2,
    event_study,
    bacon_decomposition,
    staggered_did,
    did_doubly_robust,
    placebo_test_time,
    DiDResult,
    EventStudyResult,
    BaconDecomposition,
)
from morie.iv import (
    tsls,
    wald_estimator,
    first_stage_diagnostics,
    sargan_test,
    anderson_rubin_test,
    IVResult,
    DiagnosticResult,
)
from morie.rdd import (
    sharp_rdd,
    fuzzy_rdd,
    mccrary_test,
    bandwidth_ik,
    rd_plot_data,
    RDDResult,
    BandwidthResult,
    DensityTestResult,
)
from morie.matching import (
    match_nearest_neighbor,
    estimate_propensity_score,
    balance_diagnostics,
    estimate_att_matched,
    rosenbaum_bounds,
    MatchResult,
    BalanceResult,
    TreatmentEffectResult,
)
from morie.sensitivity import (
    e_value_rr,
    e_value_or,
    rosenbaum_bounds as sensitivity_rosenbaum_bounds,
    omitted_variable_bias,
    manski_bounds,
    EValueResult,
    RosenbaumBounds as SensitivityRosenbaumBounds,
    OmittedVariableBias as OVBResult,
    tipping_point_analysis,
    TippingPointResult,
)
from morie.survival import (
    kaplan_meier,
    cox_ph,
    logrank_test,
    exponential_model,
    weibull_model,
    restricted_mean_survival_time,
    SurvivalCurve,
    CoxResult,
    LogRankResult,
    ParametricSurvivalResult,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG_SEED = 42


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(RNG_SEED)


# ===================================================================
# 1. morie.did — Difference-in-Differences
# ===================================================================


class TestDiD:
    """Tests for the morie.did module."""

    @staticmethod
    def _make_did_data(rng, n=400, true_effect=2.0):
        """Generate a 2x2 DiD panel with a known treatment effect."""
        treatment = np.repeat([0, 1], n // 2)
        post = np.tile([0, 1], n // 2)
        y = (
            1.0
            + 0.5 * treatment
            + 0.3 * post
            + true_effect * treatment * post
            + rng.normal(0, 1, n)
        )
        return pd.DataFrame({
            "outcome": y,
            "treatment": treatment,
            "post": post,
        })

    def test_did_2x2_returns_didresult(self, rng):
        df = self._make_did_data(rng)
        res = did_2x2(df, "outcome", "treatment", "post")
        assert isinstance(res, DiDResult)
        assert res.method == "did_2x2"

    def test_did_2x2_recovers_effect(self, rng):
        true_effect = 3.0
        df = self._make_did_data(rng, n=400, true_effect=true_effect)
        res = did_2x2(df, "outcome", "treatment", "post")
        assert abs(res.estimate - true_effect) < 0.5, (
            f"DiD estimate {res.estimate:.2f} far from true {true_effect}"
        )
        assert res.ci_lower < true_effect < res.ci_upper
        ci_width = res.ci_upper - res.ci_lower
        assert ci_width < 2.0, (
            f"DiD CI width {ci_width:.2f} is too wide — estimate is uninformative"
        )

    def test_did_2x2_with_covariates(self, rng):
        df = self._make_did_data(rng)
        df["x1"] = rng.normal(0, 1, len(df))
        res = did_2x2(df, "outcome", "treatment", "post", covariates=["x1"])
        assert isinstance(res, DiDResult)
        assert res.n_treated > 0
        assert res.n_control > 0

    def test_did_2x2_summary_dataframe(self, rng):
        df = self._make_did_data(rng)
        res = did_2x2(df, "outcome", "treatment", "post")
        summary = res.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "estimate" in summary.columns
        assert len(summary) == 1

    def test_did_doubly_robust_returns_didresult(self, rng):
        df = self._make_did_data(rng, n=400, true_effect=2.0)
        df["x1"] = rng.normal(0, 1, len(df))
        res = did_doubly_robust(
            df, "outcome", "treatment", "post",
            covariates=["x1"], n_bootstrap=50, seed=RNG_SEED,
        )
        assert isinstance(res, DiDResult)
        assert res.method == "did_doubly_robust"
        # Should be in the right ballpark (true effect = 2.0)
        assert abs(res.estimate - 2.0) < 1.5, (
            f"DR-DiD estimate {res.estimate:.2f} too far from true value 2.0"
        )

    def test_event_study_returns_result(self, rng):
        n_units = 40
        n_periods = 10
        treatment_time_val = 5
        records = []
        for u in range(n_units):
            g = 1 if u >= n_units // 2 else 0
            tt = treatment_time_val if g == 1 else np.inf
            for t in range(n_periods):
                post = 1 if (g == 1 and t >= treatment_time_val) else 0
                y = 1.0 + 0.5 * g + 0.1 * t + 2.0 * post + rng.normal(0, 0.5)
                records.append({
                    "unit": u, "time": t, "outcome": y,
                    "treatment_time": tt,
                })
        df = pd.DataFrame(records)
        res = event_study(
            df, "outcome", "unit", "time", "treatment_time",
            leads=3, lags=3,
        )
        assert isinstance(res, EventStudyResult)
        assert isinstance(res.coefficients, pd.DataFrame)
        assert "relative_time" in res.coefficients.columns
        assert res.reference_period == -1

    def test_bacon_decomposition_returns_result(self, rng):
        n_units = 30
        n_periods = 8
        records = []
        for u in range(n_units):
            if u < 10:
                g_time = 3
            elif u < 20:
                g_time = 5
            else:
                g_time = np.inf  # never treated
            for t in range(n_periods):
                treat = 1 if (g_time != np.inf and t >= g_time) else 0
                y = 1.0 + 0.2 * t + 1.5 * treat + rng.normal(0, 0.5)
                records.append({
                    "unit": u, "time": t, "outcome": y, "treat": treat,
                })
        df = pd.DataFrame(records)
        res = bacon_decomposition(df, "outcome", "treat", "unit", "time")
        assert isinstance(res, BaconDecomposition)
        assert isinstance(res.components, pd.DataFrame)
        assert "weight" in res.components.columns

    def test_placebo_test_time_returns_dataframe(self, rng):
        df = self._make_did_data(rng, n=400)
        # Add a time column: 0 = pre, 1 = post
        df["time"] = df["post"]
        # Expand to multiple pre-periods
        records = []
        for t in range(5):
            chunk = df.copy()
            chunk["time"] = t
            chunk["post"] = int(t >= 3)
            chunk["outcome"] = chunk["outcome"] + rng.normal(0, 0.1, len(chunk))
            records.append(chunk)
        panel = pd.concat(records, ignore_index=True)
        res = placebo_test_time(
            panel, "outcome", "treatment", "time",
            true_treatment_time=3, placebo_times=[1, 2],
        )
        assert isinstance(res, pd.DataFrame)
        assert "estimate" in res.columns

    def test_staggered_did_returns_dict(self, rng):
        n_units = 30
        n_periods = 8
        records = []
        for u in range(n_units):
            if u < 10:
                g_time = 3
            elif u < 20:
                g_time = 5
            else:
                g_time = np.inf
            for t in range(n_periods):
                treat = 1 if (g_time != np.inf and t >= g_time) else 0
                y = 1.0 + 0.2 * t + 1.5 * treat + rng.normal(0, 0.5)
                records.append({
                    "unit": u, "time": t, "outcome": y,
                    "treatment_time": g_time,
                })
        df = pd.DataFrame(records)
        res = staggered_did(
            df, "outcome", "unit", "time", "treatment_time",
            n_bootstrap=30, seed=RNG_SEED,
        )
        assert isinstance(res, dict)
        assert "group_time" in res
        assert "overall" in res


# ===================================================================
# 2. morie.iv — Instrumental Variables
# ===================================================================


class TestIV:
    """Tests for the morie.iv module."""

    @staticmethod
    def _make_iv_data(rng, n=500, true_beta=1.5):
        """Generate IV data with a known structural coefficient.

        DGP: z -> d -> y, with confounding (u affects both d and y).
        """
        z = rng.binomial(1, 0.5, n).astype(float)
        u = rng.normal(0, 1, n)
        d = 0.8 * z + 0.5 * u + rng.normal(0, 0.3, n)
        y = true_beta * d + 0.5 * u + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "d": d, "z": z})

    def test_tsls_returns_ivresult(self, rng):
        df = self._make_iv_data(rng)
        res = tsls(df, "y", ["d"], ["z"])
        assert isinstance(res, IVResult)
        assert res.method == "2sls"
        assert res.n_obs == len(df)

    def test_tsls_recovers_coefficient(self):
        """Use a strong first stage to ensure reliable recovery."""
        rng = np.random.default_rng(123)
        true_beta = 2.0
        n = 1000
        z = rng.binomial(1, 0.5, n).astype(float)
        u = rng.normal(0, 1, n)
        d = 2.0 * z + 0.3 * u + rng.normal(0, 0.3, n)  # strong first stage
        y = true_beta * d + 0.3 * u + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "d": d, "z": z})
        res = tsls(df, "y", ["d"], ["z"])
        # Find the endogenous variable coefficient by name
        endog_idx = res.variable_names.index("d")
        beta_hat = float(res.coefficients[endog_idx])
        assert abs(beta_hat - true_beta) < 1.0, (
            f"2SLS estimate {beta_hat:.2f} far from {true_beta}"
        )

    def test_tsls_with_exogenous(self, rng):
        df = self._make_iv_data(rng)
        df["w"] = rng.normal(0, 1, len(df))
        res = tsls(df, "y", ["d"], ["z"], exogenous=["w"])
        assert isinstance(res, IVResult)
        assert "w" in res.variable_names

    def test_wald_estimator_returns_ivresult(self, rng):
        df = self._make_iv_data(rng)
        # Binarize d for Wald
        df["d_bin"] = (df["d"] > df["d"].median()).astype(int)
        res = wald_estimator(df, "y", "d_bin", "z")
        assert isinstance(res, IVResult)
        assert res.method == "wald"

    def test_wald_estimator_details(self, rng):
        df = self._make_iv_data(rng)
        df["d_bin"] = (df["d"] > df["d"].median()).astype(int)
        res = wald_estimator(df, "y", "d_bin", "z")
        assert "reduced_form" in res.details
        assert "first_stage" in res.details

    def test_first_stage_diagnostics_returns_dataframe(self, rng):
        df = self._make_iv_data(rng)
        diag = first_stage_diagnostics(df, ["d"], ["z"])
        assert isinstance(diag, pd.DataFrame)
        assert "f_stat" in diag.columns
        assert "partial_r2" in diag.columns
        # With a strong instrument, F should be reasonably large
        assert diag["f_stat"].iloc[0] > 5

    def test_sargan_test_exactly_identified(self, rng):
        df = self._make_iv_data(rng)
        res = sargan_test(df, "y", ["d"], ["z"])
        assert isinstance(res, DiagnosticResult)
        # Exactly identified => test not applicable
        assert np.isnan(res.statistic) or "Exactly identified" in res.details.get("message", "")

    def test_sargan_test_overidentified(self, rng):
        df = self._make_iv_data(rng)
        # Add a second instrument
        df["z2"] = 0.3 * df["z"] + rng.normal(0, 1, len(df))
        res = sargan_test(df, "y", ["d"], ["z", "z2"])
        assert isinstance(res, DiagnosticResult)
        assert res.test_name == "sargan"
        assert 0 <= res.p_value <= 1

    def test_anderson_rubin_test_returns_result(self, rng):
        df = self._make_iv_data(rng)
        res = anderson_rubin_test(df, "y", ["d"], ["z"])
        assert isinstance(res, DiagnosticResult)
        assert res.test_name == "anderson_rubin"
        assert res.statistic >= 0

    def test_anderson_rubin_null_hypothesis(self, rng):
        """Under true null, AR test should not reject."""
        n = 300
        z = rng.binomial(1, 0.5, n).astype(float)
        u = rng.normal(0, 1, n)
        d = 0.8 * z + 0.5 * u + rng.normal(0, 0.3, n)
        # beta0 = 1.5
        true_beta = 1.5
        y = true_beta * d + 0.5 * u + rng.normal(0, 1, n)
        df = pd.DataFrame({"y": y, "d": d, "z": z})
        res = anderson_rubin_test(df, "y", ["d"], ["z"], beta0=true_beta)
        # Should not reject at 0.05
        assert res.p_value > 0.01  # generous bound


# ===================================================================
# 3. morie.rdd — Regression Discontinuity Design
# ===================================================================


class TestRDD:
    """Tests for the morie.rdd module."""

    @staticmethod
    def _make_rdd_data(rng, n=500, true_jump=3.0, cutoff=0.0):
        """Generate sharp RDD data with a known jump at the cutoff."""
        x = rng.uniform(-2, 2, n)
        treat = (x >= cutoff).astype(float)
        y = 0.5 * x + true_jump * treat + rng.normal(0, 0.5, n)
        return pd.DataFrame({"outcome": y, "running": x})

    def test_sharp_rdd_returns_rddresult(self, rng):
        df = self._make_rdd_data(rng)
        res = sharp_rdd(df, "outcome", "running", cutoff=0.0, bandwidth=1.0)
        assert isinstance(res, RDDResult)
        assert res.bandwidth == 1.0
        assert res.n_left > 0
        assert res.n_right > 0

    def test_sharp_rdd_recovers_jump(self, rng):
        true_jump = 3.0
        df = self._make_rdd_data(rng, n=500, true_jump=true_jump)
        res = sharp_rdd(df, "outcome", "running", cutoff=0.0, bandwidth=1.5)
        assert abs(res.estimate - true_jump) < 1.5, (
            f"Sharp RDD estimate {res.estimate:.2f} far from {true_jump}"
        )

    def test_sharp_rdd_auto_bandwidth(self, rng):
        df = self._make_rdd_data(rng)
        res = sharp_rdd(df, "outcome", "running", cutoff=0.0)
        assert res.bandwidth > 0
        assert not np.isnan(res.estimate)

    def test_fuzzy_rdd_returns_rddresult(self, rng):
        n = 500
        x = rng.uniform(-2, 2, n)
        # Fuzzy treatment: probability jumps at cutoff
        p_treat = np.where(x >= 0, 0.8, 0.2)
        treat = rng.binomial(1, p_treat)
        y = 0.5 * x + 3.0 * treat + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"outcome": y, "running": x, "treatment": treat})
        res = fuzzy_rdd(df, "outcome", "running", "treatment", cutoff=0.0, bandwidth=1.5)
        assert isinstance(res, RDDResult)
        assert "first_stage_jump" in res.details

    def test_mccrary_test_no_manipulation(self, rng):
        """With uniform running variable, McCrary test returns valid output."""
        x = rng.uniform(-2, 2, 500)
        res = mccrary_test(x, cutoff=0.0)
        assert isinstance(res, DensityTestResult)
        assert res.method == "mccrary"
        assert 0 <= res.p_value <= 1
        assert np.isfinite(res.statistic)

    def test_mccrary_test_with_manipulation(self, rng):
        """Stack observations just above the cutoff to simulate manipulation."""
        x_clean = rng.uniform(-2, 2, 300)
        x_manip = rng.uniform(0.0, 0.1, 200)  # pile-up above cutoff
        x = np.concatenate([x_clean, x_manip])
        res = mccrary_test(x, cutoff=0.0)
        assert isinstance(res, DensityTestResult)

    def test_bandwidth_ik_returns_result(self, rng):
        x = rng.uniform(-2, 2, 500)
        y = 0.5 * x + 3.0 * (x >= 0) + rng.normal(0, 0.5, 500)
        res = bandwidth_ik(x, y, cutoff=0.0)
        assert isinstance(res, BandwidthResult)
        assert res.h_opt > 0
        assert res.method == "IK"

    def test_rd_plot_data_returns_dict(self, rng):
        df = self._make_rdd_data(rng)
        res = rd_plot_data(df, "outcome", "running", cutoff=0.0, bandwidth=1.5)
        assert isinstance(res, dict)
        assert "binned" in res
        assert "global_poly" in res
        assert "local_poly" in res
        assert isinstance(res["binned"], pd.DataFrame)

    @pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
    def test_sharp_rdd_kernels(self, rng, kernel):
        df = self._make_rdd_data(rng)
        res = sharp_rdd(df, "outcome", "running", cutoff=0.0, bandwidth=1.5, kernel=kernel)
        assert isinstance(res, RDDResult)
        assert not np.isnan(res.estimate)


# ===================================================================
# 4. morie.matching — Matching estimators
# ===================================================================


class TestMatching:
    """Tests for the morie.matching module."""

    @staticmethod
    def _make_matching_data(rng, n=300):
        """Generate observational data for matching tests.

        DGP: treatment depends on x1, x2; outcome depends on treatment + x1.
        True ATT ~ 2.0.
        """
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        ps_true = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
        treatment = rng.binomial(1, ps_true)
        y = 1.0 + 2.0 * treatment + 0.8 * x1 + rng.normal(0, 1, n)
        return pd.DataFrame({
            "outcome": y, "treatment": treatment,
            "x1": x1, "x2": x2,
        })

    def test_estimate_propensity_score_returns_series(self, rng):
        df = self._make_matching_data(rng)
        ps = estimate_propensity_score(df, "treatment", ["x1", "x2"])
        assert isinstance(ps, pd.Series)
        assert len(ps) == len(df)
        assert (ps >= 0).all() and (ps <= 1).all()

    def test_match_nearest_neighbor_returns_matchresult(self, rng):
        df = self._make_matching_data(rng)
        res = match_nearest_neighbor(df, "treatment", ["x1", "x2"])
        assert isinstance(res, MatchResult)
        assert res.n_treated > 0
        assert res.n_matched_control > 0
        assert res.method == "nearest_neighbor"
        assert isinstance(res.match_pairs, pd.DataFrame)

    def test_match_nearest_neighbor_with_caliper(self, rng):
        df = self._make_matching_data(rng)
        res = match_nearest_neighbor(df, "treatment", ["x1", "x2"], caliper=0.5)
        assert isinstance(res, MatchResult)
        # Some units may be unmatched due to caliper
        assert res.details.get("n_unmatched", 0) >= 0

    def test_balance_diagnostics_returns_balanceresult(self, rng):
        df = self._make_matching_data(rng)
        res = balance_diagnostics(df, "treatment", ["x1", "x2"])
        assert isinstance(res, BalanceResult)
        assert isinstance(res.balance_table, pd.DataFrame)
        assert "smd" in res.balance_table.columns
        assert res.max_smd >= 0

    def test_balance_improves_after_matching(self, rng):
        df = self._make_matching_data(rng, n=400)
        bal_before = balance_diagnostics(df, "treatment", ["x1", "x2"])
        match_res = match_nearest_neighbor(df, "treatment", ["x1", "x2"])
        bal_after = balance_diagnostics(
            match_res.matched_data, "treatment", ["x1", "x2"]
        )
        # After matching, balance should improve or at least not be terrible
        assert bal_after.overall_balance <= bal_before.overall_balance + 0.2

    def test_estimate_att_matched_returns_result(self, rng):
        df = self._make_matching_data(rng)
        match_res = match_nearest_neighbor(df, "treatment", ["x1", "x2"])
        att = estimate_att_matched(
            df, "outcome", "treatment", match_res.match_pairs,
        )
        assert isinstance(att, TreatmentEffectResult)
        assert att.estimand == "ATT"
        assert att.n_obs > 0
        # True ATT is 2.0; should be in the right direction
        assert att.estimate > 0

    def test_rosenbaum_bounds_returns_dataframe(self, rng):
        df = self._make_matching_data(rng)
        match_res = match_nearest_neighbor(df, "treatment", ["x1", "x2"])
        bounds = rosenbaum_bounds(
            df, "outcome", "treatment", match_res.match_pairs,
        )
        assert isinstance(bounds, pd.DataFrame)
        assert "gamma" in bounds.columns
        assert "p_lower" in bounds.columns
        assert "p_upper" in bounds.columns
        assert len(bounds) > 0


# ===================================================================
# 5. morie.sensitivity — Sensitivity analysis
# ===================================================================


class TestSensitivity:
    """Tests for the morie.sensitivity module."""

    def test_e_value_rr_basic(self):
        res = e_value_rr(2.0)
        assert isinstance(res, EValueResult)
        assert res.rr == 2.0
        # E-value for RR=2.0 is 2 + sqrt(2*1) = 2 + sqrt(2) ~ 3.41
        expected_e = 2.0 + np.sqrt(2.0 * 1.0)
        assert abs(res.e_value_point - expected_e) < 0.01

    def test_e_value_rr_with_ci(self):
        res = e_value_rr(2.5, ci_lower=1.5, ci_upper=4.0)
        assert isinstance(res, EValueResult)
        assert res.ci_lower == 1.5
        assert res.ci_upper == 4.0
        assert res.e_value_ci > 1.0

    def test_e_value_rr_protective(self):
        """RR < 1 should still produce a valid E-value."""
        res = e_value_rr(0.5)
        assert isinstance(res, EValueResult)
        # RR=0.5 => inverted to 2.0 => same E-value as RR=2.0
        expected_e = 2.0 + np.sqrt(2.0 * 1.0)
        assert abs(res.e_value_point - expected_e) < 0.01

    def test_e_value_or_rare_outcome(self):
        """For rare outcomes, OR approximates RR."""
        res = e_value_or(2.0, prevalence=0.05)
        assert isinstance(res, EValueResult)
        # Should be close to e_value_rr(2.0)
        expected_e = 2.0 + np.sqrt(2.0 * 1.0)
        assert abs(res.e_value_point - expected_e) < 0.1

    def test_e_value_or_common_outcome(self):
        """For common outcomes, OR -> RR conversion changes the E-value."""
        res = e_value_or(2.0, prevalence=0.3)
        assert isinstance(res, EValueResult)
        # After correction, RR < OR for common outcomes
        assert res.e_value_point > 1.0

    @pytest.mark.parametrize("rr", [1.5, 2.0, 3.0, 5.0])
    def test_e_value_monotonic_in_rr(self, rr):
        """E-value should increase with RR."""
        res = e_value_rr(rr)
        assert res.e_value_point >= rr

    def test_sensitivity_rosenbaum_bounds_returns_result(self):
        rng = np.random.default_rng(RNG_SEED)
        treated = rng.normal(5, 1, 50)
        control = rng.normal(3, 1, 50)
        res = sensitivity_rosenbaum_bounds(treated, control)
        assert isinstance(res, SensitivityRosenbaumBounds)
        assert len(res.gamma_values) > 0
        assert res.critical_gamma >= 1.0

    @pytest.mark.parametrize("method", ["wilcoxon", "sign"])
    def test_sensitivity_rosenbaum_bounds_methods(self, method):
        rng = np.random.default_rng(RNG_SEED)
        treated = rng.normal(5, 1, 50)
        control = rng.normal(3, 1, 50)
        res = sensitivity_rosenbaum_bounds(treated, control, method=method)
        assert res.method == method
        assert len(res.p_upper) == len(res.gamma_values)

    def test_omitted_variable_bias_returns_result(self):
        res = omitted_variable_bias(
            estimate=0.5, se=0.1, dof=100,
            r2_yd_x=0.15, partial_r2_treatment=0.15,
        )
        assert isinstance(res, OVBResult)
        assert res.rv_q >= 0
        assert res.rv_qa >= 0
        assert "partial R2" in res.interpretation

    def test_omitted_variable_bias_with_benchmarks(self):
        res = omitted_variable_bias(
            estimate=0.5, se=0.1, dof=100,
            r2_yd_x=0.15, partial_r2_treatment=0.15,
            benchmark_covariates={"age": 0.05, "income": 0.10},
        )
        assert "age" in res.benchmark_bounds
        assert "income" in res.benchmark_bounds

    def test_manski_bounds_returns_dict(self):
        rng = np.random.default_rng(RNG_SEED)
        y1 = rng.normal(5, 1, 100)
        y0 = rng.normal(3, 1, 100)
        res = manski_bounds(y1, y0, p_treated=0.5, outcome_range=(0, 10))
        assert isinstance(res, dict)
        assert "lower_bound" in res
        assert "upper_bound" in res
        assert res["lower_bound"] <= res["upper_bound"]
        # Point estimate should be inside bounds
        assert res["lower_bound"] <= res["point_estimate"] <= res["upper_bound"]

    def test_tipping_point_analysis_returns_result(self):
        res = tipping_point_analysis(
            estimate=0.5, se=0.15, n_treated=100, n_control=100,
        )
        assert isinstance(res, TippingPointResult)
        assert res.original_estimate == 0.5
        assert len(res.delta_values) > 0
        assert len(res.adjusted_estimates) == len(res.delta_values)


# ===================================================================
# 6. morie.survival — Survival analysis
# ===================================================================


class TestSurvival:
    """Tests for the morie.survival module."""

    @staticmethod
    def _make_survival_data(rng, n=200, lam=0.1, censor_rate=0.3):
        """Generate exponential survival data with random censoring."""
        true_time = rng.exponential(1 / lam, n)
        censor_time = rng.exponential(1 / (lam * censor_rate / (1 - censor_rate + 1e-10)), n)
        time = np.minimum(true_time, censor_time)
        event = (true_time <= censor_time).astype(int)
        return time, event

    def test_kaplan_meier_returns_survivalcurve(self, rng):
        time, event = self._make_survival_data(rng)
        km = kaplan_meier(time, event)
        assert isinstance(km, SurvivalCurve)
        assert len(km.times) > 0
        assert len(km.survival) == len(km.times)
        # Survival should be monotonically non-increasing
        assert np.all(np.diff(km.survival) <= 1e-10)

    def test_kaplan_meier_starts_at_one(self, rng):
        time, event = self._make_survival_data(rng)
        km = kaplan_meier(time, event)
        # First survival probability should be close to 1
        assert km.survival[0] >= 0.9

    def test_kaplan_meier_ci_bounds(self, rng):
        time, event = self._make_survival_data(rng)
        km = kaplan_meier(time, event)
        assert np.all(km.ci_lower <= km.survival + 1e-10)
        assert np.all(km.ci_upper >= km.survival - 1e-10)
        assert np.all(km.ci_lower >= 0)
        assert np.all(km.ci_upper <= 1)

    @pytest.mark.parametrize("ci_method", ["greenwood", "log-log"])
    def test_kaplan_meier_ci_methods(self, rng, ci_method):
        time, event = self._make_survival_data(rng)
        km = kaplan_meier(time, event, ci_method=ci_method)
        assert isinstance(km, SurvivalCurve)
        assert ci_method in km.method

    def test_logrank_test_returns_result(self, rng):
        time1, event1 = self._make_survival_data(rng, n=100, lam=0.1)
        time2, event2 = self._make_survival_data(rng, n=100, lam=0.2)
        time = np.concatenate([time1, time2])
        event = np.concatenate([event1, event2])
        group = np.array([0] * 100 + [1] * 100)
        res = logrank_test(time, event, group)
        assert isinstance(res, LogRankResult)
        assert res.n_groups == 2
        assert res.test_statistic >= 0
        assert 0 <= res.p_value <= 1

    def test_logrank_detects_difference(self, rng):
        """Groups with very different hazards should yield small p-value."""
        time1, event1 = self._make_survival_data(rng, n=100, lam=0.05)
        time2, event2 = self._make_survival_data(rng, n=100, lam=0.5)
        time = np.concatenate([time1, time2])
        event = np.concatenate([event1, event2])
        group = np.array([0] * 100 + [1] * 100)
        res = logrank_test(time, event, group)
        assert res.p_value < 0.05

    def test_cox_ph_returns_coxresult(self, rng):
        n = 200
        x1 = rng.normal(0, 1, n)
        true_beta = 0.5
        lam = 0.1 * np.exp(true_beta * x1)
        true_time = rng.exponential(1 / lam)
        censor_time = rng.exponential(20, n)
        time = np.minimum(true_time, censor_time)
        event = (true_time <= censor_time).astype(int)
        df = pd.DataFrame({"time": time, "event": event, "x1": x1})
        res = cox_ph(df, "time", "event", ["x1"])
        assert isinstance(res, CoxResult)
        assert len(res.coefficients) == 1
        assert len(res.hazard_ratios) == 1
        assert res.n_observations == n

    def test_cox_ph_recovers_coefficient(self, rng):
        n = 300
        x1 = rng.normal(0, 1, n)
        true_beta = 0.8
        lam = 0.1 * np.exp(true_beta * x1)
        true_time = rng.exponential(1 / lam)
        censor_time = rng.exponential(30, n)
        time = np.minimum(true_time, censor_time)
        event = (true_time <= censor_time).astype(int)
        df = pd.DataFrame({"time": time, "event": event, "x1": x1})
        res = cox_ph(df, "time", "event", ["x1"])
        # Should recover coefficient roughly
        assert abs(res.coefficients[0] - true_beta) < 0.5, (
            f"Cox PH estimate {res.coefficients[0]:.2f} far from {true_beta}"
        )

    def test_exponential_model_returns_result(self, rng):
        time, event = self._make_survival_data(rng, lam=0.1)
        res = exponential_model(time, event)
        assert isinstance(res, ParametricSurvivalResult)
        assert res.distribution == "Exponential"
        assert "lambda" in res.coefficients
        # Lambda should be close to 0.1
        assert abs(res.coefficients["lambda"] - 0.1) < 0.05

    def test_weibull_model_returns_result(self, rng):
        time, event = self._make_survival_data(rng)
        res = weibull_model(time, event)
        assert isinstance(res, ParametricSurvivalResult)
        assert res.distribution == "Weibull"
        assert "shape" in res.coefficients
        assert "scale" in res.coefficients
        # Exponential data => shape should be close to 1
        assert abs(res.coefficients["shape"] - 1.0) < 0.5

    def test_restricted_mean_survival_time_returns_dict(self, rng):
        time, event = self._make_survival_data(rng)
        res = restricted_mean_survival_time(time, event, tau=20.0)
        assert isinstance(res, dict)
        assert "rmst" in res
        assert "se" in res
        assert "ci_lower" in res
        assert "ci_upper" in res
        assert res["rmst"] > 0
        assert res["tau"] == 20.0

    def test_rmst_auto_tau(self, rng):
        time, event = self._make_survival_data(rng)
        res = restricted_mean_survival_time(time, event)
        assert res["tau"] > 0
        assert res["rmst"] > 0

    @pytest.mark.parametrize("model_fn,dist_name", [
        (exponential_model, "Exponential"),
        (weibull_model, "Weibull"),
    ])
    def test_parametric_models_aic_bic(self, rng, model_fn, dist_name):
        time, event = self._make_survival_data(rng)
        res = model_fn(time, event)
        assert res.distribution == dist_name
        assert np.isfinite(res.aic)
        assert np.isfinite(res.bic)
        assert res.aic > 0
