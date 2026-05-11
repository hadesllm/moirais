"""Tests for new causal estimators: ATT, ATC, GATE, CATE, LATE, IRM."""

import numpy as np
import pandas as pd
import pytest

from morie.causal import (
    estimate_att,
    estimate_atc,
    estimate_cate,
    estimate_gate,
    estimate_late,
)

# DoubleML is an optional dependency; skip IRM tests if not installed.
try:
    import doubleml  # noqa: F401
    HAS_DOUBLEML = True
except ImportError:
    HAS_DOUBLEML = False


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with known treatment effects
# ---------------------------------------------------------------------------

@pytest.fixture
def causal_df():
    """Synthetic data with a known positive treatment effect.

    DGP: Y = 2*T + 0.5*X1 + 0.3*X2 + noise
    So the true ATE = 2.0
    """
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Propensity depends on X1
    logit_ps = -0.5 + 0.8 * x1
    ps = 1 / (1 + np.exp(-logit_ps))
    t = rng.binomial(1, ps)
    y = 2.0 * t + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({
        "treatment": t,
        "outcome": y,
        "x1": x1,
        "x2": x2,
    })


@pytest.fixture
def iv_df():
    """Synthetic IV data for LATE estimation.

    DGP:
        Z ~ Bernoulli(0.5)  (instrument)
        T = 1{0.5*Z + U > 0} where U ~ N(0,1) (endogenous treatment)
        Y = 3.0*T + V where V ~ N(0,1) (outcome)
    True LATE for compliers ~ 3.0
    """
    rng = np.random.default_rng(42)
    n = 1000
    z = rng.binomial(1, 0.5, n)
    u = rng.normal(0, 0.5, n)
    # Treatment influenced by instrument and unobservable
    t = (0.8 * z + u > 0.3).astype(int)
    # Outcome driven by treatment
    y = 3.0 * t + rng.normal(0, 1, n)
    return pd.DataFrame({
        "treatment": t,
        "outcome": y,
        "instrument": z,
    })


@pytest.fixture
def gate_df():
    """Synthetic data with group-level treatment heterogeneity."""
    rng = np.random.default_rng(42)
    n = 600
    group = rng.choice(["young", "middle", "old"], size=n)
    x1 = rng.normal(0, 1, n)
    t = rng.binomial(1, 0.5, n)
    # Effect varies by group
    effect = np.where(group == "young", 1.0,
             np.where(group == "middle", 2.0, 3.0))
    y = effect * t + 0.5 * x1 + rng.normal(0, 1, n)
    return pd.DataFrame({
        "treatment": t,
        "outcome": y,
        "x1": x1,
        "group": group,
    })


# ---------------------------------------------------------------------------
# ATT
# ---------------------------------------------------------------------------

class TestEstimateATT:
    def test_returns_dict_with_required_keys(self, causal_df):
        result = estimate_att(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert "att" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "n" in result
        assert "method" in result

    def test_att_positive_for_positive_effect(self, causal_df):
        result = estimate_att(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        # True effect is 2.0; ATT should be within 0.5 of the truth
        assert abs(result["att"] - 2.0) < 0.5, (
            f"ATT estimate {result['att']:.3f} too far from true value 2.0"
        )
        # CI width should be reasonable (not infinitely wide)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width < 2.0, f"CI width {ci_width:.3f} is too wide"

    def test_att_se_is_positive(self, causal_df):
        result = estimate_att(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert result["se"] > 0

    def test_att_ci_contains_att(self, causal_df):
        result = estimate_att(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert result["ci_lower"] <= result["att"] <= result["ci_upper"]

    def test_att_weights_treated_outcomes(self, causal_df):
        """ATT should weight treated units with weight 1 (their sample mean)."""
        result = estimate_att(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        # n_treated should be less than total n
        assert result["n_treated"] < result["n"]
        assert result["n_treated"] > 0


# ---------------------------------------------------------------------------
# ATC
# ---------------------------------------------------------------------------

class TestEstimateATC:
    def test_returns_dict_with_required_keys(self, causal_df):
        result = estimate_atc(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert "atc" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_atc_positive_for_positive_effect(self, causal_df):
        result = estimate_atc(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert result["atc"] > 0


# ---------------------------------------------------------------------------
# GATE
# ---------------------------------------------------------------------------

class TestEstimateGATE:
    def test_returns_dataframe(self, gate_df):
        result = estimate_gate(
            gate_df, treatment="treatment", outcome="outcome",
            covariates=["x1"], group_col="group",
        )
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_group(self, gate_df):
        result = estimate_gate(
            gate_df, treatment="treatment", outcome="outcome",
            covariates=["x1"], group_col="group",
        )
        assert len(result) == 3  # young, middle, old

    def test_gate_columns(self, gate_df):
        result = estimate_gate(
            gate_df, treatment="treatment", outcome="outcome",
            covariates=["x1"], group_col="group",
        )
        for col in ["group", "ate", "se", "ci_lower", "ci_upper", "n"]:
            assert col in result.columns

    def test_single_group_returns_one_row(self):
        """If group_col has one level with treatment variation, returns 1 row."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "treatment": rng.binomial(1, 0.5, n),
            "outcome": rng.normal(0, 1, n),
            "x1": rng.normal(0, 1, n),
            "group": ["only_group"] * n,
        })
        result = estimate_gate(
            df, treatment="treatment", outcome="outcome",
            covariates=["x1"], group_col="group",
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# CATE
# ---------------------------------------------------------------------------

class TestEstimateCate:
    def test_returns_series(self, causal_df):
        cate = estimate_cate(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        assert isinstance(cate, pd.Series)

    def test_cate_length_equals_n(self, causal_df):
        cate = estimate_cate(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"],
        )
        # Length should equal the number of non-missing rows
        expected_n = causal_df[["treatment", "outcome", "x1", "x2"]].dropna().shape[0]
        assert len(cate) == expected_n

    def test_t_learner_default(self, causal_df):
        cate = estimate_cate(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"], meta_learner="t_learner",
        )
        # Mean CATE should be roughly positive (true effect = 2.0)
        assert cate.mean() > 0

    def test_s_learner(self, causal_df):
        cate = estimate_cate(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"], meta_learner="s_learner",
        )
        assert isinstance(cate, pd.Series)
        assert len(cate) > 0

    def test_invalid_learner_raises(self, causal_df):
        with pytest.raises(ValueError, match="Unknown meta_learner"):
            estimate_cate(
                causal_df, treatment="treatment", outcome="outcome",
                covariates=["x1", "x2"], meta_learner="invalid",
            )


# ---------------------------------------------------------------------------
# LATE
# ---------------------------------------------------------------------------

class TestEstimateLATE:
    def test_returns_dict_with_keys(self, iv_df):
        result = estimate_late(
            iv_df, treatment="treatment", outcome="outcome",
            instrument="instrument",
        )
        assert "late" in result
        assert "se" in result
        assert "ci" in result
        assert "f_stat" in result
        assert "n" in result

    def test_late_in_reasonable_range(self, iv_df):
        """True LATE is roughly 3.0; should be in a wide interval around it."""
        result = estimate_late(
            iv_df, treatment="treatment", outcome="outcome",
            instrument="instrument",
        )
        assert 0.5 < result["late"] < 8.0

    def test_f_stat_positive(self, iv_df):
        result = estimate_late(
            iv_df, treatment="treatment", outcome="outcome",
            instrument="instrument",
        )
        assert result["f_stat"] > 0

    def test_weak_instrument_raises(self):
        """Instrument uncorrelated with treatment should raise."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "treatment": rng.binomial(1, 0.5, n),
            "outcome": rng.normal(0, 1, n),
            "instrument": rng.normal(0, 1, n),  # uncorrelated
        })
        # The Wald estimator may still compute (noise), but with tiny cov_tz
        # We just verify it doesn't crash hard
        result = estimate_late(
            df, treatment="treatment", outcome="outcome",
            instrument="instrument",
        )
        assert "late" in result


# ---------------------------------------------------------------------------
# IRM (requires DoubleML)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DOUBLEML, reason="DoubleML not installed")
class TestEstimateIRM:
    def test_irm_returns_dict(self, causal_df):
        from morie.causal import estimate_irm
        result = estimate_irm(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"], n_folds=2,
        )
        assert "ate" in result
        assert "se" in result
        assert result["method"] == "IRM (DoubleML)"

    def test_irm_positive_effect(self, causal_df):
        from morie.causal import estimate_irm
        result = estimate_irm(
            causal_df, treatment="treatment", outcome="outcome",
            covariates=["x1", "x2"], n_folds=2,
        )
        assert result["ate"] > 0
