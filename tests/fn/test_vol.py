"""Tests for moirais.fn.vol — regional volatility metric."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.vol import volat as vol
from moirais.fn._containers import VolRes


@pytest.fixture()
def placement_df():
    """Synthetic placement data with region columns."""
    rng = np.random.default_rng(42)
    regions = ["Central", "Eastern", "Northern", "Toronto", "Western"]
    n = 80
    return pd.DataFrame({
        "unique_individual_id": [f"P{i % 20:04d}" for i in range(n)],
        "end_fiscal_year": rng.choice([2020, 2021], n),
        "region_at_time_of_placement": rng.choice(regions, n),
        "region_most_recent_placement": rng.choice(regions, n),
    })


class TestVolat:
    """Tests for regional volatility."""

    def test_returns_volres(self, placement_df):
        """Should return VolRes dataclass."""
        result = vol(placement_df)
        assert isinstance(result, VolRes)

    def test_mean_positive(self, placement_df):
        """Mean volatility should be positive (at least 1 region each)."""
        result = vol(placement_df)
        assert result.mean > 0

    def test_single_region_person(self):
        """Person placed in only one region should have volatility = 1."""
        df = pd.DataFrame({
            "unique_individual_id": ["P0001", "P0001"],
            "end_fiscal_year": [2021, 2021],
            "region_at_time_of_placement": ["Central", "Central"],
            "region_most_recent_placement": ["Central", "Central"],
        })
        result = vol(df)
        assert result.mean == pytest.approx(1.0)

    def test_data_has_vm_column(self, placement_df):
        """Result data should contain a 'vm' column."""
        result = vol(placement_df)
        assert "vm" in result.data.columns
