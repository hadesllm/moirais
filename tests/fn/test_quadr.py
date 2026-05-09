"""Tests for moirais.fn.quadr — quadrat test."""
import numpy as np
import pytest
from moirais.fn.quadr import quadrat_test


class TestQuadratTest:
    def test_uniform_points(self):
        pts = np.random.default_rng(42).uniform(0, 10, (100, 2))
        res = quadrat_test(pts, n_quadrats=4)
        assert 0 <= res.p_value <= 1

    def test_too_few_quadrats_raises(self):
        with pytest.raises(ValueError):
            quadrat_test(np.ones((5, 2)), n_quadrats=1)
