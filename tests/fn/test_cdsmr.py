"""Tests for moirais.fn.cdsmr -- standardized mortality ratio."""

import pytest
from moirais.fn.cdsmr import standardized_mortality_ratio


class TestSMR:
    def test_basic(self):
        res = standardized_mortality_ratio(observed=120, expected=100)
        assert res.estimate == pytest.approx(1.2)

    def test_ci(self):
        res = standardized_mortality_ratio(120, 100)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            standardized_mortality_ratio(10, 0)
