"""Tests for moirais.fn.crent — cross-entropy."""
import numpy as np
import pytest
from moirais.fn.crent import cross_entropy


class TestCrossEntropy:
    def test_same_is_entropy(self):
        p = np.array([0.5, 0.5])
        res = cross_entropy(p, p)
        expected = -np.sum(p * np.log(p))
        assert res.estimate == pytest.approx(expected, abs=1e-10)

    def test_worse_prediction_higher(self):
        p = np.array([0.9, 0.1])
        q_good = np.array([0.8, 0.2])
        q_bad = np.array([0.2, 0.8])
        ce_good = cross_entropy(p, q_good)
        ce_bad = cross_entropy(p, q_bad)
        assert ce_bad.estimate > ce_good.estimate
