"""Tests for morie.fn.splgn -- NASA-TLX."""

from morie.fn.splgn import nasa_tlx, splgn
from morie.fn._containers import DescriptiveResult


class TestSplgn:
    def test_alias(self):
        assert splgn is nasa_tlx

    def test_all_50(self):
        result = nasa_tlx()
        assert isinstance(result, DescriptiveResult)
        assert result.value == 50.0

    def test_weighted(self):
        result = nasa_tlx(mental=80, physical=20, weights={"mental": 5, "physical": 1})
        assert result.value > 50
