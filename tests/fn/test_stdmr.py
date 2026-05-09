"""Tests for moirais.fn.stdmr -- Standardized mortality ratio."""

import pytest
from moirais.fn.stdmr import standardized_mortality_ratio


class TestSMR:
    def test_equal(self):
        res = standardized_mortality_ratio(observed=50, expected=50.0)
        assert res.estimate == pytest.approx(1.0)

    def test_excess(self):
        res = standardized_mortality_ratio(observed=100, expected=50.0)
        assert res.estimate == pytest.approx(2.0)

    def test_ci_brackets(self):
        res = standardized_mortality_ratio(observed=50, expected=50.0)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_invalid(self):
        with pytest.raises(ValueError):
            standardized_mortality_ratio(observed=10, expected=0)
