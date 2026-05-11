"""Tests for morie.fn.sudur -- substance duration."""

import pytest
from morie.fn.sudur import substance_duration


class TestSubstanceDuration:
    def test_basic(self):
        res = substance_duration([2, 5, 8, 12, 3])
        assert res.name == "substance_duration"
        assert res.value == pytest.approx(6.0)

    def test_chronic(self):
        res = substance_duration([1, 2, 6, 11, 15])
        assert res.extra["pct_chronic_5yr"] == pytest.approx(60.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            substance_duration([])
