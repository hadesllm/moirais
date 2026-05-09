"""Tests for moirais.fn.suhdnk -- heavy drinking."""

import pytest
from moirais.fn.suhdnk import heavy_drinking


class TestHeavyDrinking:
    def test_basic(self):
        res = heavy_drinking([1, 2, 5, 6, 8, 3])
        assert res.measure == "heavy_drinking_prevalence"
        assert res.estimate == pytest.approx(3 / 6)

    def test_none_heavy(self):
        res = heavy_drinking([1, 2, 3, 4])
        assert res.estimate == 0.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            heavy_drinking([])
