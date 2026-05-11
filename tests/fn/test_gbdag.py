"""Tests for morie.fn.gbdag -- GBD age pattern."""

from morie.fn.gbdag import gbd_age_pattern


class TestGBDAgePattern:
    def test_basic(self):
        res = gbd_age_pattern({"0-14": 100, "15-49": 300, "50-69": 400, "70+": 200})
        assert res.extra["peak_age_group"] == "50-69"

    def test_empty(self):
        import pytest
        with pytest.raises(ValueError):
            gbd_age_pattern({})
