"""Tests for morie.fn.rpl — regional placement analysis."""

import pytest
import numpy as np
import pandas as pd
from morie.fn.rpl import rplace as rpl
from morie.fn._containers import RplRes


@pytest.fixture()
def otis_df():
    """Synthetic OTIS-like correctional placement data."""
    rng = np.random.default_rng(42)
    n = 200
    regions = ["Central", "Eastern", "Northern", "Toronto", "Western"]
    ages = ["18 to 24", "25 to 49", "50+"]
    return pd.DataFrame({
        "unique_individual_id": [f"P{i:04d}" for i in range(n)],
        "age_category": rng.choice(ages, n),
        "region_at_time_of_placement": rng.choice(regions, n),
        "end_fiscal_year": rng.choice([2020, 2021], n),
        "gender": rng.choice(["Male", "Female"], n),
    })


class TestRplace:
    """Tests for regional placement analysis."""

    def test_returns_rplres(self, otis_df):
        """Should return an RplRes dataclass."""
        result = rpl(otis_df, year=2021)
        assert isinstance(result, RplRes)

    def test_props_rows_sum_to_one(self, otis_df):
        """Proportion rows should sum to approximately 1."""
        result = rpl(otis_df, year=2021)
        row_sums = result.props.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0, abs=1e-10)

    def test_sex_filter(self, otis_df):
        """Filtering by sex should reduce counts."""
        all_result = rpl(otis_df, year=2021)
        male_result = rpl(otis_df, year=2021, sex="Male")
        assert male_result.counts.sum().sum() <= all_result.counts.sum().sum()

    def test_year_metadata(self, otis_df):
        """Year should be stored in result."""
        result = rpl(otis_df, year=2020)
        assert result.year == 2020
