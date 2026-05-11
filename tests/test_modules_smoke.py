"""Smoke tests for previously untested modules.

Each test verifies a module imports and its key function runs on synthetic
data without crashing. Assertions check return types and attributes
derived from the actual result objects — no hardcoded values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_df():
    """Small synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "outcome": np.random.randn(n) + np.random.choice([0, 1], n) * 0.5,
        "treatment": np.random.choice([0, 1], n),
        "post": np.random.choice([0, 1], n),
        "running": np.random.randn(n),
        "time": np.tile(np.arange(20), 10),
        "unit": np.repeat(np.arange(10), 20),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
        "event": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "weight": np.random.uniform(0.5, 2.0, n),
    })


class TestDiD:
    def test_did_2x2_returns_result(self, synthetic_df):
        from morie.did import did_2x2
        result = did_2x2(synthetic_df, "outcome", "treatment", "post")
        # Check actual attributes from DiDResult dataclass
        assert hasattr(result, "estimate"), "DiDResult must have estimate"
        assert hasattr(result, "p_value"), "DiDResult must have p_value"
        assert isinstance(result.estimate, float)
        assert 0 <= result.p_value <= 1


class TestRDD:
    def test_sharp_rdd_returns_result(self, synthetic_df):
        from morie.rdd import sharp_rdd
        result = sharp_rdd(synthetic_df, "outcome", "running", cutoff=0.0)
        assert hasattr(result, "estimate"), "RDDResult must have estimate"
        assert isinstance(result.estimate, float)


class TestSurvival:
    def test_kaplan_meier_returns_result(self, synthetic_df):
        from morie.survival import kaplan_meier
        time_vals = np.abs(synthetic_df["running"].values) + 0.1
        event_vals = synthetic_df["event"].values
        result = kaplan_meier(time_vals, event_vals)
        # Check it has survival data
        assert result is not None
        assert hasattr(result, "survival_times") or hasattr(result, "times") or hasattr(result, "survival_probabilities")


class TestMatching:
    def test_nearest_neighbor_matching_runs(self, synthetic_df):
        from morie.matching import match_nearest_neighbor
        result = match_nearest_neighbor(
            synthetic_df, "treatment", ["x1", "x2"]
        )
        assert result is not None
        assert hasattr(result, "n_matched") or hasattr(result, "matched_data")


class TestMissing:
    def test_littles_mcar_returns_result(self, synthetic_df):
        from morie.missing import littles_mcar_test
        df = synthetic_df[["x1", "x2", "x3"]].copy()
        # Introduce some missing values
        mask = np.random.random(len(df)) < 0.1
        df.loc[mask, "x1"] = np.nan
        result = littles_mcar_test(df)
        assert hasattr(result, "test_statistic")
        assert hasattr(result, "p_value")
        assert isinstance(result.p_value, float)


class TestBootstrapMethods:
    def test_bootstrap_ci_bounds(self):
        from morie.bootstrap_methods import bootstrap
        np.random.seed(42)
        data = np.random.randn(100)
        result = bootstrap(data, np.mean, n_boot=500, ci_method="percentile")
        assert hasattr(result, "estimate")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert result.ci_lower <= result.estimate <= result.ci_upper


class TestMultipleTesting:
    def test_bh_correction_output(self):
        from morie.multiple_testing import benjamini_hochberg
        p_values = np.array([0.01, 0.04, 0.03, 0.20, 0.50])
        result = benjamini_hochberg(p_values)
        assert hasattr(result, "adjusted")
        assert hasattr(result, "rejected")
        assert len(result.adjusted) == len(p_values)
        # Adjusted p-values should be >= original
        for orig, adj in zip(p_values, result.adjusted):
            assert adj >= orig or abs(adj - orig) < 1e-10


class TestEffectSizes:
    def test_cohens_d_returns_estimate(self):
        from morie.effect_sizes import cohens_d
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50) + 0.5
        result = cohens_d(x, y)
        assert hasattr(result, "estimate")
        assert isinstance(result.estimate, float)

    def test_hedges_g_returns_estimate(self):
        from morie.effect_sizes import hedges_g
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50) + 0.5
        result = hedges_g(x, y)
        assert hasattr(result, "estimate")


class TestSensitivity:
    def test_e_value_positive(self):
        from morie.sensitivity import e_value_rr
        result = e_value_rr(2.5, ci_lower=1.1)
        assert hasattr(result, "e_value_point")
        assert result.e_value_point > 1.0, "E-value must exceed 1 for RR > 1"
        assert hasattr(result, "interpretation")


class TestDiagnostics:
    def test_collinearity_diagnostics_runs(self, synthetic_df):
        from morie.diagnostics import collinearity_diagnostics
        X = synthetic_df[["x1", "x2", "x3"]].values
        result = collinearity_diagnostics(X, column_names=["x1", "x2", "x3"])
        assert hasattr(result, "vif")
        assert len(result.vif) == 3


class TestStatistics:
    def test_one_sample_ttest_detects_shift(self):
        from morie.statistics import one_sample_ttest
        np.random.seed(42)
        data = np.random.randn(50) + 1.0  # shifted mean
        result = one_sample_ttest(data, mu0=0)
        assert hasattr(result, "test_statistic")
        assert hasattr(result, "p_value")
        assert result.p_value < 0.05, "Should reject H0 (mean ≠ 0)"

    def test_pearson_detects_correlation(self):
        from morie.statistics import pearson_correlation
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5  # strong positive correlation
        result = pearson_correlation(x, y)
        assert result.test_statistic > 0.5, "Strong correlation expected"
