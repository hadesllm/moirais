"""Tests for morie.fn.smote — SMOTE oversampling with fallback."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.smote import apply_smote as smote


@pytest.fixture()
def imbalanced_data():
    """Imbalanced binary dataset: 90 majority, 10 minority."""
    rng = np.random.default_rng(42)
    n_maj, n_min = 90, 10
    X = pd.DataFrame(rng.standard_normal((n_maj + n_min, 3)), columns=["a", "b", "c"])
    y = pd.Series([0] * n_maj + [1] * n_min, name="y")
    return X, y


class TestApplySmote:
    """Tests for apply_smote."""

    def test_returns_tuple_of_three(self, imbalanced_data):
        """Should return (DataFrame, Series, dict)."""
        X, y = imbalanced_data
        result = smote(X, y)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.Series)
        assert isinstance(result[2], dict)

    def test_resampled_has_more_rows(self, imbalanced_data):
        """Resampled data should have >= original rows."""
        X, y = imbalanced_data
        X_res, y_res, _ = smote(X, y)
        assert len(X_res) >= len(X)
        assert len(y_res) >= len(y)

    def test_status_has_method_key(self, imbalanced_data):
        """Status dict must contain a 'method' key."""
        X, y = imbalanced_data
        _, _, status = smote(X, y)
        assert "method" in status
        assert status["method"] in ("smote", "random_oversample")

    def test_random_fallback_works(self):
        """With very few minority samples, should fall back to random oversample."""
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.standard_normal((102, 2)), columns=["x1", "x2"])
        y = pd.Series([0] * 100 + [1] * 2, name="y")
        X_res, y_res, status = smote(X, y)
        # Only 2 minority samples — SMOTE needs k_neighbors > minority count
        assert len(X_res) >= len(X)
        assert status["total_after"] >= status["total_before"]
