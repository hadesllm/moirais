"""Tests for morie.dataset — dataset-agnostic profiling and analysis engine."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morie.dataset import (
    ColumnProfile,
    DatasetProfile,
    MeasurementLevel,
    infer_measurement_level,
    load_dataset,
    profile_dataset,
    suggest_analysis_plan,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def epi_df():
    """Synthetic epidemiological DataFrame with known variable types."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "id": range(n),
        "treatment_arm": rng.choice([0, 1], size=n),
        "outcome_event": rng.normal(5.0, 2.0, size=n),
        "age_group": rng.choice(["18-24", "25-34", "35-44", "45-54"], size=n),
        "gender": rng.choice(["Male", "Female"], size=n),
        "province_region": rng.choice(["ON", "QC", "BC", "AB", "SK", "MB", "NS",
                                        "NB", "NL", "PE", "YT"], size=n),
        "weight": rng.uniform(0.5, 4.0, size=n),
        "income": rng.exponential(50000, size=n),
        "year_of_birth": rng.integers(1960, 2005, size=n),
        "likert_satisfaction": rng.integers(1, 6, size=n),  # 1-5 scale
        "count_visits": rng.poisson(3, size=n),
    })


@pytest.fixture
def minimal_binary_df():
    """Minimal DataFrame with clear binary treatment and binary outcome."""
    return pd.DataFrame({
        "drug_exposure": [0, 1, 0, 1, 0, 1, 0, 1],
        "death_event": [0, 0, 1, 1, 0, 1, 0, 1],
        "age": [20, 30, 40, 50, 25, 35, 45, 55],
        "bmi": [22.0, 25.0, 28.0, 30.0, 21.5, 24.5, 27.5, 31.0],
    })


# ---------------------------------------------------------------------------
# infer_measurement_level
# ---------------------------------------------------------------------------

class TestInferMeasurementLevel:
    """Test NOIR classification for individual columns."""

    def test_nominal_string_column(self):
        s = pd.Series(["A", "B", "C", "A", "B"], name="category")
        assert infer_measurement_level(s) == MeasurementLevel.NOMINAL

    def test_nominal_binary_int(self):
        s = pd.Series([0, 1, 0, 1, 0], name="flag")
        assert infer_measurement_level(s) == MeasurementLevel.NOMINAL

    def test_nominal_boolean(self):
        s = pd.Series([True, False, True, False], name="is_active")
        assert infer_measurement_level(s) == MeasurementLevel.NOMINAL

    def test_ordinal_named_string(self):
        s = pd.Series(["low", "med", "high", "low", "med"], name="severity_level")
        assert infer_measurement_level(s) == MeasurementLevel.ORDINAL

    def test_ordinal_named_int(self):
        s = pd.Series([1, 2, 3, 4, 5], name="likert_q1")
        assert infer_measurement_level(s) == MeasurementLevel.ORDINAL

    def test_ordinal_age_group_string(self):
        s = pd.Series(["18-24", "25-34", "35-44"], name="age_group")
        assert infer_measurement_level(s) == MeasurementLevel.ORDINAL

    def test_interval_year(self):
        s = pd.Series([2018.0, 2019.0, 2020.0, 2021.0], name="year")
        assert infer_measurement_level(s) == MeasurementLevel.INTERVAL

    def test_interval_score(self):
        s = pd.Series([1.5, 2.3, -0.5, 3.1], name="z_score")
        assert infer_measurement_level(s) == MeasurementLevel.INTERVAL

    def test_ratio_positive_float(self):
        s = pd.Series([10.5, 20.3, 30.1, 0.5], name="income")
        assert infer_measurement_level(s) == MeasurementLevel.RATIO

    def test_ratio_count(self):
        s = pd.Series([0, 1, 5, 10, 3], name="n_visits")
        assert infer_measurement_level(s) == MeasurementLevel.RATIO

    def test_interval_negative_int(self):
        s = pd.Series([-5, -2, 0, 3, 7], name="temperature_delta")
        assert infer_measurement_level(s) == MeasurementLevel.INTERVAL

    def test_high_cardinality_nominal(self):
        """String column with > ordinal_threshold uniques stays NOMINAL."""
        s = pd.Series([f"cat_{i}" for i in range(50)], name="many_cats")
        assert infer_measurement_level(s, ordinal_threshold=10) == MeasurementLevel.NOMINAL


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------

class TestProfileDataset:
    """Test the full profiling pipeline."""

    def test_basic_profile_shape(self, epi_df):
        profile = profile_dataset(epi_df)
        assert profile.n_rows == 200
        assert profile.n_cols == 11
        assert len(profile.columns) == 11

    def test_detects_treatment(self, epi_df):
        profile = profile_dataset(epi_df)
        assert profile.suggested_treatment == "treatment_arm"

    def test_detects_outcome(self, epi_df):
        profile = profile_dataset(epi_df)
        assert profile.suggested_outcome == "outcome_event"

    def test_detects_weight(self, epi_df):
        profile = profile_dataset(epi_df)
        assert profile.suggested_weights == "weight"

    def test_hint_overrides_detection(self, epi_df):
        profile = profile_dataset(
            epi_df,
            hint_treatment="age_group",
            hint_outcome="income",
            hint_weights="count_visits",
        )
        assert profile.suggested_treatment == "age_group"
        assert profile.suggested_outcome == "income"
        assert profile.suggested_weights == "count_visits"

    def test_column_profiles_populated(self, epi_df):
        profile = profile_dataset(epi_df)
        for cp in profile.columns.values():
            assert isinstance(cp, ColumnProfile)
            assert cp.name in epi_df.columns
            assert isinstance(cp.level, MeasurementLevel)
            assert cp.missing_pct >= 0.0

    def test_binary_detection(self, minimal_binary_df):
        profile = profile_dataset(minimal_binary_df)
        assert profile.columns["drug_exposure"].is_binary

    def test_constant_detection(self):
        df = pd.DataFrame({"const": [1, 1, 1], "x": [1.0, 2.0, 3.0]})
        profile = profile_dataset(df)
        assert profile.columns["const"].is_constant

    def test_to_dict_serializable(self, epi_df):
        profile = profile_dataset(epi_df)
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["n_rows"] == 200
        assert "columns" in d

    def test_summary_table_returns_string(self, epi_df):
        profile = profile_dataset(epi_df)
        table = profile.summary_table()
        assert isinstance(table, str)
        assert "200" in table

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="at least one row"):
            profile_dataset(pd.DataFrame())

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError, match="Expected pandas DataFrame"):
            profile_dataset([1, 2, 3])

    def test_missing_percentage(self):
        df = pd.DataFrame({
            "complete": [1, 2, 3, 4],
            "half_missing": [1.0, np.nan, 3.0, np.nan],
        })
        profile = profile_dataset(df)
        assert profile.columns["complete"].missing_pct == 0.0
        assert abs(profile.columns["half_missing"].missing_pct - 50.0) < 0.1


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Test file ingestion."""

    def test_load_csv(self, epi_df):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            epi_df.to_csv(f.name, index=False)
            loaded = load_dataset(f.name)
        assert loaded.shape == epi_df.shape

    def test_load_tsv(self, epi_df):
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
            epi_df.to_csv(f.name, sep="\t", index=False)
            loaded = load_dataset(f.name)
        assert loaded.shape == epi_df.shape

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/data.csv")

    def test_raises_on_bad_extension(self, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_dataset(bad_file)


# ---------------------------------------------------------------------------
# suggest_analysis_plan
# ---------------------------------------------------------------------------

class TestSuggestAnalysisPlan:
    """Test analysis plan suggestions."""

    def test_always_includes_descriptive(self, epi_df):
        profile = profile_dataset(epi_df)
        plan = suggest_analysis_plan(profile)
        assert len(plan) >= 1
        assert plan[0]["analysis"] == "descriptive_profile"

    def test_suggests_ipw_with_binary_treatment(self, minimal_binary_df):
        profile = profile_dataset(minimal_binary_df)
        plan = suggest_analysis_plan(profile)
        analyses = [s["analysis"] for s in plan]
        # Should suggest at least propensity/IPW since we have binary treatment + outcome
        assert any("ipw" in a or "propensity" in a for a in analyses)

    def test_suggests_survey_weights(self, epi_df):
        profile = profile_dataset(epi_df)
        plan = suggest_analysis_plan(profile)
        analyses = [s["analysis"] for s in plan]
        assert "survey_weighted_estimates" in analyses

    def test_plan_entries_have_required_keys(self, epi_df):
        profile = profile_dataset(epi_df)
        plan = suggest_analysis_plan(profile)
        for step in plan:
            assert "analysis" in step
            assert "rationale" in step
            assert "required_vars" in step
