"""Tests for moirais.fn.daly -- Disability-Adjusted Life Years."""

import pytest
from moirais.fn.daly import disability_adjusted_life_years


class TestDALY:
    def test_sum(self):
        result = disability_adjusted_life_years(100.0, 50.0)
        assert result["daly"] == pytest.approx(150.0)

    def test_zero_yld(self):
        result = disability_adjusted_life_years(80.0, 0.0)
        assert result["daly"] == pytest.approx(80.0)

    def test_components_preserved(self):
        result = disability_adjusted_life_years(42.0, 18.0)
        assert result["yll"] == pytest.approx(42.0)
        assert result["yld"] == pytest.approx(18.0)

    def test_negative_yll_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            disability_adjusted_life_years(-10.0, 5.0)
