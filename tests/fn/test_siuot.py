"""Tests for morie.fn.siuot — SIU outcome."""

import pytest
from morie.fn.siuot import siu_outcome
from morie.fn._containers import DescriptiveResult


class TestSiuOutcome:
    def test_basic(self):
        r = siu_outcome(["No charges"] * 80 + ["Charges laid"] * 20)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["proportions"]["No charges"] == pytest.approx(0.8)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            siu_outcome([])
