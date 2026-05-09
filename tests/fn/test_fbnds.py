"""Test filter_bounds (fbnds)."""
import pytest

from moirais.fn.fbnds import filter_bounds, fbnds
from moirais.fn._containers import DescriptiveResult


class TestFilterBounds:
    def test_basic(self):
        result = filter_bounds(10, 100.0, 1000.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "filter_bounds"
        assert result.value == 50.0

    def test_pass_stop(self):
        result = filter_bounds(10, 100.0, 1000.0)
        assert result.extra["f_pass"] == 75.0
        assert result.extra["f_stop"] == 125.0

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            filter_bounds(0, 100.0, 1000.0)

    def test_alias(self):
        assert fbnds is filter_bounds
