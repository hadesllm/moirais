"""Tests for moirais.fn.mhors -- Mantel-Haenszel OR."""

import pytest
from moirais.fn.mhors import mantel_haenszel_or


class TestMHOR:
    def test_single_table(self):
        res = mantel_haenszel_or([(20, 80, 10, 90)])
        assert res.measure == "OR_MH"
        assert res.estimate == pytest.approx(20 * 90 / (80 * 10), rel=0.01)

    def test_two_tables(self):
        res = mantel_haenszel_or([(20, 80, 10, 90), (30, 70, 15, 85)])
        assert res.estimate > 0

    def test_empty(self):
        with pytest.raises(ValueError):
            mantel_haenszel_or([])
