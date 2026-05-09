"""Tests for moirais.fn.epi_c -- epidemic curve."""

import numpy as np
import pandas as pd
import pytest
from moirais.fn.epi_c import epidemic_curve


class TestEpiCurve:
    def test_sum_equals_total(self):
        """Bins should sum to total events."""
        rng = np.random.default_rng(42)
        base = np.datetime64("2020-01-01")
        dates = base + rng.integers(0, 100, size=100).astype("timedelta64[D]")
        res = epidemic_curve(dates, bin_width="week")
        assert res.name == "Epidemic curve"
        assert isinstance(res.value, pd.DataFrame)
        assert res.value["count"].sum() == 100

    def test_daily_bins(self):
        """Daily bins should have at most 1 per unique date."""
        dates = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
        res = epidemic_curve(dates, bin_width="day")
        assert len(res.value) == 2

    def test_invalid_bin_raises(self):
        """Invalid bin_width should raise."""
        with pytest.raises(ValueError):
            epidemic_curve(["2020-01-01"], bin_width="year")
