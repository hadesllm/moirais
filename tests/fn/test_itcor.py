"""Tests for morie.fn.itcor — corrected item-total correlations."""

import pytest
import numpy as np
import pandas as pd
from morie.fn import itcor


@pytest.fixture()
def items_df():
    """5 items x 80 respondents."""
    rng = np.random.default_rng(42)
    latent = rng.standard_normal(80)
    data = np.column_stack([
        latent + rng.standard_normal(80) * 0.5
        for _ in range(5)
    ])
    return pd.DataFrame(data, columns=[f"q{i}" for i in range(5)])


class TestItcor:
    """Tests for corrected item-total correlations."""

    def test_returns_dataframe_with_columns(self, items_df):
        """Result should have columns item, r_total, r_corr."""
        result = itcor(items_df)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"item", "r_total", "r_corr"}

    def test_corrected_less_than_uncorrected(self, items_df):
        """Corrected r should be less than uncorrected r (part-whole)."""
        result = itcor(items_df)
        for _, row in result.iterrows():
            assert row["r_corr"] < row["r_total"]

    def test_item_names_preserved(self, items_df):
        """Item names should match DataFrame columns."""
        result = itcor(items_df)
        assert list(result["item"]) == [f"q{i}" for i in range(5)]

    def test_numpy_input(self):
        """Should work with ndarray input (auto-generates names)."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 3))
        result = itcor(data)
        assert len(result) == 3
        assert result["item"].iloc[0] == "i0"
