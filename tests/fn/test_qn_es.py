"""Tests for qn_estimator."""
import numpy as np, pytest
from moirais.fn.qn_es import qn_estimator


class TestQnEstimator:
    def test_normal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        r = qn_estimator(x)
        assert r.measure == "qn_estimator"
        assert r.estimate > 0

    def test_two_values(self):
        r = qn_estimator([1.0, 5.0])
        assert r.estimate > 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            qn_estimator([1.0])
