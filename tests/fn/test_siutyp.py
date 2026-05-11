"""Tests for morie.fn.siutyp — SIU by type."""

import pytest
from morie.fn.siutyp import siu_by_type
from morie.fn._containers import DescriptiveResult


class TestSiuByType:
    def test_basic(self):
        r = siu_by_type(["Shooting"] * 5 + ["Custody death"] * 3 + ["Sexual assault"] * 2)
        assert isinstance(r, DescriptiveResult)
        assert r.extra["counts"]["Shooting"] == 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            siu_by_type([])
