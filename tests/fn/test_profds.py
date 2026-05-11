"""Tests for morie.fn.profds — dataset profiling."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.profds import profds, profile_dataset
from morie.dataset import DatasetProfile


def test_alias_is_same_function():
    """profds and profile_dataset are the same object."""
    assert profds is profile_dataset


@pytest.fixture()
def sample_df():
    """Synthetic dataset with treatment, outcome, weight, covariate."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "treatment": rng.choice([0, 1], n),
        "outcome": rng.standard_normal(n),
        "age": rng.integers(18, 80, n),
        "survey_wt": rng.uniform(0.5, 2.0, n),
        "province": rng.choice(["ON", "QC", "BC", "AB"], n),
    })


def test_returns_dataset_profile(sample_df):
    """profile_dataset returns a DatasetProfile instance."""
    result = profds(sample_df)
    assert isinstance(result, DatasetProfile)


def test_shape_matches(sample_df):
    """n_rows and n_cols match the DataFrame."""
    result = profds(sample_df)
    assert result.n_rows == len(sample_df)
    assert result.n_cols == len(sample_df.columns)


def test_all_columns_profiled(sample_df):
    """Every column appears in the profile."""
    result = profds(sample_df)
    assert set(result.columns.keys()) == set(sample_df.columns)


def test_detects_treatment(sample_df):
    """Auto-detects the treatment column."""
    result = profds(sample_df)
    assert result.suggested_treatment == "treatment"


def test_detects_outcome(sample_df):
    """Auto-detects the outcome column."""
    result = profds(sample_df)
    assert result.suggested_outcome == "outcome"


def test_detects_weights(sample_df):
    """Auto-detects the survey weight column."""
    result = profds(sample_df)
    assert result.suggested_weights == "survey_wt"


def test_hint_overrides_detection(sample_df):
    """User hint overrides heuristic detection."""
    result = profds(sample_df, hint_treatment="age")
    assert result.suggested_treatment == "age"


def test_rejects_non_dataframe():
    """Raises TypeError for non-DataFrame input."""
    with pytest.raises(TypeError):
        profds([1, 2, 3])


def test_rejects_empty_dataframe():
    """Raises ValueError for empty DataFrame."""
    with pytest.raises(ValueError):
        profds(pd.DataFrame())


def test_to_dict(sample_df):
    """to_dict returns a serializable dictionary."""
    result = profds(sample_df)
    d = result.to_dict()
    assert d["n_rows"] == len(sample_df)
    assert "columns" in d
