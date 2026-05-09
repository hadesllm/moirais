"""Tests for moirais.fn.otd — OTIS descriptive statistics."""

import pytest
import numpy as np
import pandas as pd
from moirais.fn.otd import otdesc as otd


@pytest.fixture()
def desc_df():
    """Synthetic OTIS data for descriptive statistics."""
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame({
        "unique_individual_id": [f"P{i % 40:04d}" for i in range(n)],
        "end_fiscal_year": rng.choice([2020, 2021], n),
        "region_at_time_of_placement": rng.choice(
            ["Central", "Eastern", "Western"], n,
        ),
        "age_category": rng.choice(["18 to 24", "25 to 49", "50+"], n),
        "gender": rng.choice(["Male", "Female"], n),
    })


class TestOtdesc:
    """Tests for OTIS descriptive statistics."""

    def test_returns_dict(self, desc_df):
        """Should return a dict."""
        result = otd(desc_df)
        assert isinstance(result, dict)

    def test_has_n_total(self, desc_df):
        """Result should have n_total key."""
        result = otd(desc_df)
        assert "n_total" in result

    def test_n_total_positive(self, desc_df):
        """n_total should be positive."""
        result = otd(desc_df)
        assert result["n_total"] > 0

    def test_n_total_le_n_records(self, desc_df):
        """n_total (unique individuals) should be <= n_records."""
        result = otd(desc_df)
        assert result["n_total"] <= result["n_records"]

    def test_has_optional_breakdowns(self, desc_df):
        """Should have region, age, gender breakdowns when columns exist."""
        result = otd(desc_df)
        assert "n_by_region" in result
        assert "n_by_age" in result
        assert "n_by_gender" in result
