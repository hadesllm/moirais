"""Tests for intraclass_correlation."""
import numpy as np, pytest
from moirais.fn.icc_ import intraclass_correlation

class TestICC:
    def test_perfect_agreement(self):
        ratings = np.array([[1,1],[2,2],[3,3],[4,4]], dtype=float)
        r = intraclass_correlation(ratings)
        assert r.extra["icc_3_1"] == pytest.approx(1.0, abs=0.01)

    def test_low_agreement(self):
        rng = np.random.default_rng(0)
        ratings = rng.normal(0, 1, (10, 3))
        r = intraclass_correlation(ratings)
        assert r.extra["icc_1_1"] < 0.5
