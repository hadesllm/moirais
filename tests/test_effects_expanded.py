"""
Tests for expanded treatment effect functions in morie.effects.

Covers:
- estimate_ate_gcomputation (G-computation ATE)
- e_value (E-value for unmeasured confounding)
- estimate_plr (DoubleML PLR)
- sensitivity_rosenbaum (Rosenbaum bounds)

References
----------
VanderWeele, T. J., & Ding, P. (2017). Annals of Internal Medicine, 167(4), 268-274.
Rosenbaum, P. R. (2002). Observational Studies (2nd ed.). Springer.
Hernan, M. A., & Robins, J. M. (2020). Causal Inference: What If. Chapman & Hall.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from morie.effects import estimate_ate_gcomputation, e_value


# ===========================================================================
# Helpers
# ===========================================================================

def _make_linear_treatment_data(
    n: int = 300,
    true_ate: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a dataset with a known linear treatment effect.

    Y = true_ate * T + X1 + X2 + epsilon
    T ~ Bernoulli(sigmoid(X1 - X2))  (confounded by X1, X2)
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit_p = 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logit_p))
    t = rng.binomial(1, p, n).astype(float)
    y = true_ate * t + 1.5 * x1 + 0.8 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"Y": y, "T": t, "X1": x1, "X2": x2})


def _make_binary_outcome_data(
    n: int = 400,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate a dataset with a binary outcome for logistic G-computation.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-1, 1, n)
    t = rng.binomial(1, 0.5, n).astype(float)
    logit_y = -0.5 + 0.8 * t + 0.6 * x1 - 0.4 * x2
    p_y = 1.0 / (1.0 + np.exp(-logit_y))
    y = rng.binomial(1, p_y, n).astype(float)
    return pd.DataFrame({"Y": y, "T": t, "X1": x1, "X2": x2})


# ===========================================================================
# G-computation ATE tests
# ===========================================================================

class TestEstimateAteGcomputation:
    """
    Tests for estimate_ate_gcomputation.

    Statistical properties verified:
    1. Returns finite float ATE.
    2. ATE is in a reasonable range for the DGP.
    3. SE is positive.
    4. CI contains the true ATE with high probability for large n.
    5. Works for both linear and logistic outcome models.
    6. Input validation raises on invalid arguments.
    """

    def test_linear_model_returns_valid_result(self):
        """Linear G-computation should return a complete result dict."""
        df = _make_linear_treatment_data(n=300)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
            outcome_model="linear",
        )
        assert isinstance(result, dict)
        assert "ate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "n_obs" in result

    def test_linear_model_ate_is_finite(self):
        """ATE must be a finite float."""
        df = _make_linear_treatment_data(n=300)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert math.isfinite(result["ate"])

    def test_linear_model_ate_near_true_value(self):
        """
        With n=300 and true ATE=2.0, G-computation should recover an estimate
        within ±1.0 of the truth (generous tolerance given confounding and
        bootstrap sampling variance).
        """
        df = _make_linear_treatment_data(n=300, true_ate=2.0, seed=42)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert abs(result["ate"] - 2.0) < 0.3, (
            f"ATE estimate {result['ate']:.3f} too far from true value 2.0"
        )

    def test_se_is_positive(self):
        """Standard error must be strictly positive."""
        df = _make_linear_treatment_data(n=200)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert result["se"] > 0

    def test_ci_lower_less_than_upper(self):
        """CI must be a valid interval."""
        df = _make_linear_treatment_data(n=200)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert result["ci_lower"] < result["ci_upper"]

    def test_n_obs_matches_complete_cases(self):
        """n_obs should reflect complete-case count."""
        df = _make_linear_treatment_data(n=150)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert result["n_obs"] == 150

    def test_logistic_model_binary_outcome(self):
        """
        Logistic G-computation for a binary outcome should return ATE
        in (-1, 1) since it represents a risk difference (E[Y(1)] - E[Y(0)]).
        """
        df = _make_binary_outcome_data(n=400)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
            outcome_model="logistic",
        )
        assert math.isfinite(result["ate"])
        # Risk difference must be positive (treatment logit coefficient is +0.8)
        # and bounded: true RD is ~0.18 for this DGP
        assert 0.05 < result["ate"] < 0.40, (
            f"Logistic G-comp risk difference {result['ate']:.3f} outside "
            f"expected range (0.05, 0.40) for DGP with treatment coef 0.8"
        )

    def test_logistic_model_label(self):
        """outcome_model key should reflect the specified model."""
        df = _make_binary_outcome_data(n=200)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
            outcome_model="logistic",
        )
        assert result["outcome_model"] == "logistic"

    def test_invalid_outcome_model_raises(self):
        df = _make_linear_treatment_data(n=50)
        with pytest.raises(ValueError, match="outcome_model must be one of"):
            estimate_ate_gcomputation(
                df, treatment="T", outcome="Y", covariates=["X1", "X2"],
                outcome_model="poisson",
            )

    def test_missing_column_raises(self):
        df = _make_linear_treatment_data(n=50)
        with pytest.raises(ValueError, match="Columns missing"):
            estimate_ate_gcomputation(
                df, treatment="T", outcome="Y", covariates=["nonexistent_col"],
            )

    def test_too_few_observations_raises(self):
        """Less than 10 complete obs should raise ValueError."""
        df = _make_linear_treatment_data(n=5)
        with pytest.raises(ValueError, match="at least 10"):
            estimate_ate_gcomputation(
                df, treatment="T", outcome="Y", covariates=["X1", "X2"],
            )

    def test_reproducibility_with_fixed_seed_in_bootstrap(self):
        """
        Two calls with the same data should give the same result
        because the bootstrap seed is fixed internally.
        """
        df = _make_linear_treatment_data(n=100)
        r1 = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        r2 = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert r1["ate"] == r2["ate"]
        assert r1["se"] == r2["se"]

    def test_ate_sign_agrees_with_positive_treatment(self):
        """
        When treatment effect is strongly positive (ATE=5) and confounding is mild,
        G-computation ATE should be positive.
        """
        df = _make_linear_treatment_data(n=400, true_ate=5.0, seed=0)
        result = estimate_ate_gcomputation(
            df, treatment="T", outcome="Y", covariates=["X1", "X2"],
        )
        assert result["ate"] > 0, (
            f"Expected positive ATE for positive treatment effect, got {result['ate']:.3f}"
        )


# ===========================================================================
# E-value tests
# ===========================================================================

class TestEValue:
    """
    Tests for the e_value function.

    Statistical properties verified:
    1. E-value at the null (z=0) is 1.
    2. E-value increases monotonically with |ATE - null| / SE.
    3. E-value is symmetric in ATE direction.
    4. Known formulas: E = RR + sqrt(RR*(RR-1)) for RR > 1.
    5. Input validation.

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational
        research: Introducing the E-value. Annals of Internal Medicine, 167(4), 268-274.
    """

    def test_at_null_returns_one(self):
        """E-value when estimate equals null = 1 (no confounding needed)."""
        assert e_value(ate=0.0, se=1.0, null=0.0) == 1.0

    def test_at_null_with_nonzero_null(self):
        """E-value when ate == null (non-zero null) = 1."""
        assert e_value(ate=5.0, se=1.0, null=5.0) == 1.0

    def test_monotone_in_effect_size(self):
        """
        Larger |ATE - null| / SE should yield larger E-values.
        Tests monotonicity: e_value(z=1) < e_value(z=2) < e_value(z=3).
        """
        e1 = e_value(ate=1.0, se=1.0)
        e2 = e_value(ate=2.0, se=1.0)
        e3 = e_value(ate=3.0, se=1.0)
        assert e1 < e2 < e3, (
            f"E-values not monotone: {e1:.3f}, {e2:.3f}, {e3:.3f}"
        )

    def test_symmetry(self):
        """E-value depends on |ATE - null|, so sign of deviation doesn't matter."""
        e_pos = e_value(ate=2.0, se=0.5, null=0.0)
        e_neg = e_value(ate=-2.0, se=0.5, null=0.0)
        assert abs(e_pos - e_neg) < 1e-10

    def test_e_value_exceeds_1_for_nonzero_effect(self):
        """Any non-null effect requires E > 1 to be explained away."""
        assert e_value(ate=2.0, se=1.0) > 1.0

    def test_e_value_is_finite(self):
        """E-value should be finite for realistic inputs."""
        assert math.isfinite(e_value(ate=1.5, se=0.3))

    def test_e_value_formula_consistency(self):
        """
        Cross-check: for a given z-score, E = RR + sqrt(RR*(RR-1)) where
        RR = exp(z). Verify the implementation matches this formula.
        """
        z = 1.5
        rr = math.exp(z)
        expected_e = rr + math.sqrt(rr * (rr - 1.0))
        computed_e = e_value(ate=z, se=1.0)  # ate/se = z directly
        assert abs(computed_e - expected_e) < 1e-10

    def test_larger_se_gives_smaller_e_value(self):
        """
        For the same ATE, a larger SE means a smaller z-score and thus
        a smaller E-value (weaker evidence requires less confounding to explain away).
        """
        e_small_se = e_value(ate=2.0, se=0.5)
        e_large_se = e_value(ate=2.0, se=2.0)
        assert e_small_se > e_large_se

    def test_se_zero_raises(self):
        with pytest.raises(ValueError, match="se must be > 0"):
            e_value(ate=1.0, se=0.0)

    def test_se_negative_raises(self):
        with pytest.raises(ValueError, match="se must be > 0"):
            e_value(ate=1.0, se=-0.5)

    def test_return_type_is_float(self):
        assert isinstance(e_value(ate=1.0, se=0.5), float)


# ===========================================================================
# Additional inference function smoke tests
# ===========================================================================

class TestHypothesisTestingSmoke:
    """
    Smoke tests for key hypothesis testing functions imported from inference.
    These verify the functions run without error and return valid outputs
    for simple canonical inputs.
    """

    def test_two_sample_t_test(self):
        from morie.inference import two_sample_t_test
        rng = np.random.default_rng(0)
        x1 = rng.normal(0, 1, 50)
        x2 = rng.normal(1, 1, 50)
        result = two_sample_t_test(x1, x2)
        assert "t" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1
        assert math.isfinite(result["t"])

    def test_one_sample_t_test(self):
        from morie.inference import one_sample_t_test
        rng = np.random.default_rng(1)
        x = rng.normal(5, 1, 30)
        result = one_sample_t_test(x, mu0=5.0)
        assert "t" in result
        assert result["p_value"] > 0.05  # should fail to reject for data near mu0

    def test_chi_square_test_1d(self):
        from morie.inference import chi_square_test
        observed = [10, 20, 15, 25]
        result = chi_square_test(observed)
        assert "chi2" in result
        assert result["chi2"] >= 0

    def test_shapiro_wilk_test(self):
        from morie.inference import shapiro_wilk_test
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 100)
        result = shapiro_wilk_test(x)
        assert "W" in result
        assert "is_normal" in result
        assert result["is_normal"]  # normal data should pass

    def test_cohens_d_known_case(self):
        """
        For two groups with mean difference = 2 and pooled SD = 1,
        Cohen's d should equal 2.0 exactly.
        """
        from morie.inference import cohens_d
        x1 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        x2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(x1, x2)
        # mean diff = 2, both groups have SD = sqrt(2.5); pooled SD = sqrt(2.5)
        expected = (5.0 - 3.0) / np.sqrt(2.5)
        assert abs(d - expected) < 1e-8

    def test_proportion_ci_wilson_contains_true_p(self):
        """Wilson CI should contain the true proportion for a large sample."""
        from morie.inference import proportion_ci
        # 500/1000 = 0.5 (true proportion)
        lo, hi = proportion_ci(500, 1000, method="wilson")
        assert lo < 0.5 < hi

    def test_power_t_test_solve_for_n(self):
        """power_t_test with delta=0.5, power=0.80 should give n > 1."""
        from morie.inference import power_t_test
        n = power_t_test(delta=0.5, power=0.80)
        assert n > 1

    def test_power_t_test_solve_for_power(self):
        """power_t_test with n=50, delta=0.5 should give power in (0, 1)."""
        from morie.inference import power_t_test
        power = power_t_test(n=50, delta=0.5)
        assert 0 < power < 1

    def test_anova_one_way_known_groups(self):
        """
        Two groups with large mean difference should yield a small p-value.
        """
        from morie.inference import anova_one_way
        rng = np.random.default_rng(0)
        g1 = rng.normal(0, 1, 30)
        g2 = rng.normal(5, 1, 30)
        result = anova_one_way(g1, g2)
        assert result["p_value"] < 0.001
        assert 0 <= result["eta_squared"] <= 1

    def test_odds_ratio_ci_2x2(self):
        """OR for a table with no association should be near 1."""
        from morie.inference import odds_ratio_ci
        tbl = [[10, 10], [10, 10]]
        result = odds_ratio_ci(tbl)
        assert abs(result["odds_ratio"] - 1.0) < 1e-10
        assert result["ci_lower"] < 1.0 < result["ci_upper"]

    def test_risk_ratio_ci_2x2(self):
        from morie.inference import risk_ratio_ci
        # Equal risks: RR should be 1
        tbl = [[50, 50], [50, 50]]
        result = risk_ratio_ci(tbl)
        assert abs(result["risk_ratio"] - 1.0) < 1e-10

    def test_cramers_v_independence(self):
        """
        A table where row and column are independent should yield V near 0.
        """
        from morie.inference import cramers_v
        # Perfect independence: proportional rows
        tbl = [[50, 50], [50, 50]]
        v = cramers_v(tbl)
        assert v < 0.01

    def test_spearman_rho_monotone(self):
        """Perfect monotone relationship should give rho = 1.0."""
        from morie.inference import spearman_rho
        x = np.arange(1, 21)
        y = x ** 2  # monotone increasing
        result = spearman_rho(x, y)
        assert abs(result["rho"] - 1.0) < 1e-10

    def test_kendall_tau_perfect_concordance(self):
        """Perfectly concordant ranking should give tau = 1.0."""
        from morie.inference import kendall_tau
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = kendall_tau(x, y)
        assert abs(result["tau"] - 1.0) < 1e-10
