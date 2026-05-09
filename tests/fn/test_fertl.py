"""Tests for moirais.fn.fertl -- total fertility rate."""

import pytest
from moirais.fn.fertl import fertility_rate


class TestFertilityRate:
    def test_basic(self):
        asfr = [0.02, 0.08, 0.10, 0.08, 0.04, 0.01, 0.005]
        res = fertility_rate(asfr, interval=5)
        assert res.measure == "TFR"
        assert res.estimate == pytest.approx(sum(asfr) * 5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fertility_rate([])
