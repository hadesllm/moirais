"""Tests for moirais.fn.dtw — dynamic time warping."""
import numpy as np
import pytest
from moirais.fn.dtw import dtw_distance


class TestDTW:
    def test_identical(self):
        x = np.array([1.0, 2.0, 3.0])
        res = dtw_distance(x, x)
        assert res.estimate == pytest.approx(0.0, abs=1e-10)

    def test_different(self):
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([1.0, 0.0, 1.0])
        res = dtw_distance(x, y)
        assert res.estimate > 0
