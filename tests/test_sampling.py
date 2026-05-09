"""Tests for moirais.sampling — survey sampling and resampling utilities."""

import numpy as np
import pandas as pd
import pytest

from moirais.sampling import (
    bootstrap_sample,
    cluster_sample,
    compute_design_weights,
    design_effect,
    effective_sample_size,
    jackknife_estimate,
    pps_sample,
    simple_random_sample,
    stratified_sample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def population_df():
    """A synthetic population frame with 500 rows."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "id": range(n),
        "stratum": rng.choice(["A", "B", "C"], size=n),
        "cluster": rng.choice(range(1, 11), size=n),
        "size_measure": rng.exponential(100, size=n),
        "y": rng.normal(50, 10, size=n),
    })


@pytest.fixture
def small_df():
    """Small DataFrame for deterministic tests."""
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


# ---------------------------------------------------------------------------
# simple_random_sample
# ---------------------------------------------------------------------------

class TestSimpleRandomSample:
    def test_correct_size(self, population_df):
        sample = simple_random_sample(population_df, 50, seed=0)
        assert len(sample) == 50

    def test_reproducible(self, population_df):
        s1 = simple_random_sample(population_df, 20, seed=123)
        s2 = simple_random_sample(population_df, 20, seed=123)
        pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))

    def test_with_replacement(self, population_df):
        sample = simple_random_sample(population_df, 1000, replace=True, seed=0)
        assert len(sample) == 1000

    def test_raises_when_n_exceeds_size(self, population_df):
        with pytest.raises(ValueError, match="Cannot draw"):
            simple_random_sample(population_df, 999, replace=False)


# ---------------------------------------------------------------------------
# stratified_sample
# ---------------------------------------------------------------------------

class TestStratifiedSample:
    def test_fixed_allocation(self, population_df):
        sample = stratified_sample(population_df, "stratum", 10, seed=0)
        sizes = sample.groupby("stratum").size()
        assert all(s == 10 for s in sizes.values)

    def test_dict_allocation(self, population_df):
        alloc = {"A": 5, "B": 10, "C": 15}
        sample = stratified_sample(population_df, "stratum", alloc, seed=0)
        sizes = sample.groupby("stratum").size().to_dict()
        assert sizes == alloc

    def test_proportional_allocation(self, population_df):
        sample = stratified_sample(
            population_df, "stratum", 100, proportional=True, seed=0
        )
        assert len(sample) == 100

    def test_raises_when_stratum_too_small(self):
        df = pd.DataFrame({"s": ["X"] * 3, "v": [1, 2, 3]})
        with pytest.raises(ValueError, match="exceeds population"):
            stratified_sample(df, "s", 5)


# ---------------------------------------------------------------------------
# cluster_sample
# ---------------------------------------------------------------------------

class TestClusterSample:
    def test_correct_cluster_count(self, population_df):
        sample = cluster_sample(population_df, "cluster", 3, seed=0)
        assert sample["cluster"].nunique() == 3

    def test_all_units_within_clusters(self, population_df):
        sample = cluster_sample(population_df, "cluster", 2, seed=0)
        selected_clusters = sample["cluster"].unique()
        for c in selected_clusters:
            expected = population_df[population_df["cluster"] == c]
            actual = sample[sample["cluster"] == c]
            assert len(actual) == len(expected)

    def test_raises_when_too_many_clusters(self, population_df):
        with pytest.raises(ValueError, match="only"):
            cluster_sample(population_df, "cluster", 999)


# ---------------------------------------------------------------------------
# pps_sample
# ---------------------------------------------------------------------------

class TestPPSSample:
    def test_correct_size(self, population_df):
        sample = pps_sample(population_df, "size_measure", 20, seed=0)
        assert len(sample) == 20

    def test_raises_on_no_positive(self):
        df = pd.DataFrame({"s": [-1, -2, -3], "v": [1, 2, 3]})
        with pytest.raises(ValueError, match="No positive values"):
            pps_sample(df, "s", 1)


# ---------------------------------------------------------------------------
# bootstrap_sample
# ---------------------------------------------------------------------------

class TestBootstrapSample:
    def test_returns_correct_keys(self, small_df):
        result = bootstrap_sample(
            small_df, 100, statistic=lambda d: d["x"].mean(), seed=0
        )
        assert "mean" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "distribution" in result

    def test_distribution_length(self, small_df):
        result = bootstrap_sample(
            small_df, 500, statistic=lambda d: d["x"].mean(), seed=0
        )
        assert len(result["distribution"]) == 500

    def test_mean_close_to_population(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(10, 1, 500)})
        result = bootstrap_sample(
            df, 1000, statistic=lambda d: d["x"].mean(), seed=0
        )
        assert abs(result["mean"] - 10.0) < 0.5

    def test_ci_contains_estimate(self, small_df):
        result = bootstrap_sample(
            small_df, 500, statistic=lambda d: d["x"].mean(), seed=0
        )
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


# ---------------------------------------------------------------------------
# jackknife_estimate
# ---------------------------------------------------------------------------

class TestJackknifeEstimate:
    def test_correct_estimate(self, small_df):
        result = jackknife_estimate(small_df, statistic=lambda d: d["x"].mean())
        assert abs(result["estimate"] - 3.0) < 1e-10

    def test_se_positive(self, small_df):
        result = jackknife_estimate(small_df, statistic=lambda d: d["x"].mean())
        assert result["se"] > 0

    def test_bias_near_zero_for_mean(self, small_df):
        result = jackknife_estimate(small_df, statistic=lambda d: d["x"].mean())
        assert abs(result["bias"]) < 1e-10


# ---------------------------------------------------------------------------
# compute_design_weights
# ---------------------------------------------------------------------------

class TestComputeDesignWeights:
    def test_correct_weights(self):
        df = pd.DataFrame({"stratum": ["A"] * 10 + ["B"] * 20})
        w = compute_design_weights(df, "stratum", {"A": 1000, "B": 2000})
        assert float(w[df["stratum"] == "A"].iloc[0]) == 100.0
        assert float(w[df["stratum"] == "B"].iloc[0]) == 100.0

    def test_raises_on_missing_stratum(self):
        df = pd.DataFrame({"stratum": ["A", "B"]})
        with pytest.raises(KeyError, match="not found"):
            compute_design_weights(df, "stratum", {"A": 100})


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:
    def test_equal_weights(self):
        w = np.ones(100)
        assert effective_sample_size(w) == 100.0

    def test_unequal_weights_less_than_n(self):
        w = np.array([1.0, 1.0, 10.0])
        ess = effective_sample_size(w)
        assert ess < 3.0

    def test_empty_returns_zero(self):
        assert effective_sample_size(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# design_effect
# ---------------------------------------------------------------------------

class TestDesignEffect:
    def test_equal_weights_deff_one(self):
        w = np.ones(50)
        assert abs(design_effect(w) - 1.0) < 1e-10

    def test_unequal_weights_deff_greater_one(self):
        w = np.array([1.0, 1.0, 1.0, 10.0])
        assert design_effect(w) > 1.0

    def test_empty_returns_one(self):
        assert design_effect(np.array([])) == 1.0
