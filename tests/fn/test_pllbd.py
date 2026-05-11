"""Test pll_bandwidth (pllbd)."""
import pytest

from morie.fn.pllbd import pll_bandwidth, pllbd
from morie.fn._containers import DescriptiveResult


class TestPllBandwidth:
    def test_basic(self):
        result = pll_bandwidth(100.0, 10.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "pll_bandwidth"
        assert result.value > 0

    def test_invalid_freq(self):
        with pytest.raises(ValueError):
            pll_bandwidth(100.0, -1.0)

    def test_alias(self):
        assert pllbd is pll_bandwidth
