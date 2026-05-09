"""Tests for moirais.fn.idisc — item discrimination index."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn import idisc


@pytest.fixture()
def items_data():
    """5 items x 100 respondents with varying difficulty."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(100)
    data = np.column_stack([
        (latent + rng.standard_normal(100) * 0.3 > 0).astype(float)
        for _ in range(5)
    ])
    return pd.DataFrame(data, columns=[f"q{i}" for i in range(5)])


class TestIdisc:
    """Tests for item discrimination D-statistic."""

    def test_returns_dataframe(self, items_data):
        """Should return a DataFrame with item and d columns."""
        result = idisc(items_data)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"item", "d"}

    def test_d_values_bounded(self, items_data):
        """Discrimination index should be between -1 and 1."""
        result = idisc(items_data)
        assert (result["d"] >= -1).all()
        assert (result["d"] <= 1).all()

    def test_length_matches_items(self, items_data):
        """Should return one row per item."""
        result = idisc(items_data)
        assert len(result) == 5

    def test_numpy_input(self):
        """Should work with ndarray input."""
        rng = np.random.default_rng(42)
        data = (rng.standard_normal((80, 3)) > 0).astype(float)
        result = idisc(data)
        assert len(result) == 3
        assert result["item"].iloc[0] == "i0"
