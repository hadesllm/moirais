"""Tests for morie.fn.gbdcm -- GBD compare."""

import pytest
from morie.fn.gbdcm import gbd_compare


class TestGBDCompare:
    def test_basic(self):
        res = gbd_compare({"heart": 500, "stroke": 300, "diabetes": 200})
        assert res.extra["top_condition"] == "heart"
        assert res.extra["top_pct"] == pytest.approx(50.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            gbd_compare({})
