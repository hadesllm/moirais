"""Tests for morie.fn.rct — restrictive confinement trends."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.rct import rctrnd as rct


@pytest.fixture()
def trend_df():
    """Synthetic multi-year placement data."""
    rng = np.random.default_rng(42)
    regions = ["Central", "Eastern", "Northern"]
    n = 150
    return pd.DataFrame({
        "unique_individual_id": [f"P{i:04d}" for i in range(n)],
        "end_fiscal_year": rng.choice([2019, 2020, 2021], n),
        "region_at_time_of_placement": rng.choice(regions, n),
    })


class TestRctrnd:
    """Tests for restrictive confinement trend analysis."""

    def test_returns_dataframe(self, trend_df):
        """Should return a DataFrame."""
        result = rct(trend_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, trend_df):
        """Should have year, region, n_individuals, n_placements columns."""
        result = rct(trend_df)
        assert set(result.columns) == {"year", "region", "n_individuals", "n_placements"}

    def test_n_individuals_positive(self, trend_df):
        """All n_individuals counts should be positive."""
        result = rct(trend_df)
        assert (result["n_individuals"] > 0).all()

    def test_sorted_by_year_region(self, trend_df):
        """Result should be sorted by year then region."""
        result = rct(trend_df)
        years = result["year"].tolist()
        assert years == sorted(years)
