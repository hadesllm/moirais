"""Test bandwidth_limit (bwlmt)."""
import pytest

from moirais.fn.bwlmt import bandwidth_limit, bwlmt
from moirais.fn._containers import DescriptiveResult


class TestBandwidthLimit:
    def test_basic(self):
        result = bandwidth_limit(1000.0, 400.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "bandwidth_limit"
        assert result.value == 500.0

    def test_no_aliasing(self):
        result = bandwidth_limit(1000.0, 400.0)
        assert result.extra["aliased"] is False
        assert result.extra["margin"] == 100.0

    def test_aliasing(self):
        result = bandwidth_limit(1000.0, 600.0)
        assert result.extra["aliased"] is True

    def test_invalid_fs(self):
        with pytest.raises(ValueError):
            bandwidth_limit(-1.0, 100.0)

    def test_alias(self):
        assert bwlmt is bandwidth_limit
