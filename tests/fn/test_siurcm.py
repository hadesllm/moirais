"""Tests for moirais.fn.siurcm — SIU recommendation."""

import pytest
from moirais.fn.siurcm import siu_recommendation
from moirais.fn._containers import DescriptiveResult


class TestSiuRecommendation:
    def test_basic(self):
        r = siu_recommendation(["No charges"] * 70 + ["Charges"] * 30)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["proportions"]["Charges"] == pytest.approx(0.3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            siu_recommendation([])
