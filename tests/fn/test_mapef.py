"""Tests for morie.fn.mapef -- MAPE."""
import numpy as np
import pytest
from morie.fn.mapef import mape


class TestMAPE:
    def test_perfect(self):
        a = np.array([10.0, 20.0, 30.0])
        res = mape(a, a)
        assert res.value == 0.0

    def test_basic(self):
        a = np.array([10.0, 20.0, 30.0])
        f = np.array([11.0, 19.0, 33.0])
        res = mape(a, f)
        assert res.value > 0

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            mape(np.zeros(5), np.ones(5))

    def test_cheatsheet(self):
        from morie.fn.mapef import cheatsheet
        assert isinstance(cheatsheet(), str)
