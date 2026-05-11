"""Tests for morie.fn.adel — alpha if item deleted."""

import pytest
import numpy as np
import pandas as pd
from morie.fn import adel


@pytest.fixture()
def items_5():
    """5-item scale, 60 respondents."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(60)
    data = np.column_stack([
        latent + rng.standard_normal(60) * 0.4
        for _ in range(5)
    ])
    return pd.DataFrame(data, columns=[f"q{i}" for i in range(5)])


class TestAdel:
    """Tests for alpha-if-deleted."""

    def test_returns_dataframe_with_columns(self, items_5):
        """Result should have columns item and adel."""
        result = adel(items_5)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"item", "adel"}

    def test_length_equals_k(self, items_5):
        """Should return one row per item (k rows, not k-1)."""
        result = adel(items_5)
        assert len(result) == 5

    def test_adel_values_finite(self, items_5):
        """All alpha-if-deleted values should be finite."""
        result = adel(items_5)
        assert result["adel"].notna().all()
        assert np.all(np.isfinite(result["adel"].values))

    def test_numpy_input(self):
        """Should work with ndarray input."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((40, 4))
        result = adel(data)
        assert len(result) == 4
