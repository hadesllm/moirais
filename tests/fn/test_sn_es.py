"""Tests for sn_estimator."""
import numpy as np, pytest
from morie.fn.sn_es import sn_estimator


class TestSnEstimator:
    def test_normal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = sn_estimator(x)
        assert r.measure == "sn_estimator"
        assert r.estimate == pytest.approx(1.0, abs=0.3)

    def test_positive(self):
        r = sn_estimator([1, 2, 3, 4, 5])
        assert r.estimate > 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            sn_estimator([1.0])
