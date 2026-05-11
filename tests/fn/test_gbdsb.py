"""Tests for morie.fn.gbdsb -- GBD subgroup."""

import pytest
from morie.fn.gbdsb import gbd_subgroup


class TestGBDSubgroup:
    def test_basic(self):
        res = gbd_subgroup(
            dalys=[100, 200, 150, 250],
            groups=["M", "M", "F", "F"],
        )
        assert res.extra["n_groups"] == 2
        assert res.extra["total_dalys"] == pytest.approx(700.0)


def test_cheatsheet():
    from morie.fn.gbdsb import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
