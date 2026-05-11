"""Tests for morie.fn.siuof — SIU officer force."""

import pytest
from morie.fn.siuof import siu_officer_force
from morie.fn._containers import DescriptiveResult


class TestSiuOfficerForce:
    def test_basic(self):
        r = siu_officer_force(["Firearm"] * 5 + ["CEW"] * 10 + ["Physical"] * 15)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["counts"]["CEW"] == 10

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            siu_officer_force([])
