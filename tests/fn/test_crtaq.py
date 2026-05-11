"""Tests for morie.fn.crtaq — acquittal rate."""

import pytest
from morie.fn.crtaq import court_acquittal
from morie.fn._containers import DescriptiveResult


class TestAcquittal:
    def test_basic(self):
        outcomes = ["Convicted", "Acquitted", "Convicted", "Acquitted"]
        offenses = ["Assault", "Assault", "Theft", "Theft"]
        r = court_acquittal(outcomes, offenses)
        assert isinstance(r, DescriptiveResult)
        assert r.value == pytest.approx(0.5)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            court_acquittal(["A"], ["B", "C"])
