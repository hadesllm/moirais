"""Tests for morie.fn.siucmp — SIU comparison."""

import pytest
from morie.fn.siucmp import siu_comparison
from morie.fn._containers import DescriptiveResult


class TestSiuComparison:
    def test_basic(self):
        r = siu_comparison({"Ontario": 5.0, "BC": 3.0, "Alberta": 4.0})
        assert isinstance(r, DescriptiveResult)
        assert r.extra["highest"] == "Ontario"
        assert r.extra["lowest"] == "BC"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            siu_comparison({})
