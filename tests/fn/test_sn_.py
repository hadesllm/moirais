"""Tests for sn_estimator."""
import numpy as np, pytest
from morie.fn.sn_ import sn_estimator

class TestSn:
    def test_normal(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        r = sn_estimator(x)
        assert r.estimate == pytest.approx(1.0, abs=0.3)

    def test_small(self):
        x = np.array([1.0, 2.0, 3.0])
        r = sn_estimator(x)
        assert r.estimate > 0
