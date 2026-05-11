"""Tests for morie.fn.suinit -- initiation age."""

import pytest
from morie.fn.suinit import initiation_age


class TestInitiationAge:
    def test_basic(self):
        res = initiation_age([14, 16, 18, 21, 25])
        assert res.name == "initiation_age"
        assert res.value == pytest.approx(18.0)

    def test_pct_under_18(self):
        res = initiation_age([12, 14, 16, 20, 22])
        assert res.extra["pct_under_18"] == pytest.approx(60.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            initiation_age([])
