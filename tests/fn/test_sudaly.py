"""Tests for morie.fn.sudaly -- substance DALY."""

import pytest
from morie.fn.sudaly import substance_daly


class TestSubstanceDaly:
    def test_basic(self):
        res = substance_daly(yll=100, yld=50)
        assert res.estimate == pytest.approx(150.0)
        assert res.extra["pct_yll"] == pytest.approx(100 / 150 * 100)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            substance_daly(yll=-1, yld=10)
