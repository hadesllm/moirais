"""Tests for moirais.fn.le -- life expectancy."""

import numpy as np
import pytest
from moirais.fn.le import life_expectancy


class TestLifeExpectancy:
    def test_basic(self):
        """3-group life table should produce positive e0."""
        age_starts = np.array([0, 15, 45])
        deaths = np.array([50, 30, 200])
        population = np.array([10000, 20000, 15000])
        res = life_expectancy(age_starts, deaths, population)
        assert res.name == "Life expectancy"
        assert res.value > 0

    def test_extra_arrays(self):
        """Extra dict should contain life table columns."""
        age_starts = np.array([0, 15, 45])
        deaths = np.array([50, 30, 200])
        population = np.array([10000, 20000, 15000])
        res = life_expectancy(age_starts, deaths, population)
        assert "lx" in res.extra
        assert "ex" in res.extra
        assert res.extra["lx"][0] == pytest.approx(100_000.0)

    def test_too_few_groups_raises(self):
        """Single age group should raise."""
        with pytest.raises(ValueError):
            life_expectancy(np.array([0]), np.array([10]), np.array([100]))
