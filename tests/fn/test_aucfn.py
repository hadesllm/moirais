"""Test auc_compute (aucfn)."""
import numpy as np
from moirais.fn.aucfn import auc_compute, aucfn
from moirais.fn._containers import DescriptiveResult


class TestAucfn:
    def test_basic(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        result = auc_compute(y_true, y_scores)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "auc_compute"
        assert result.extra["auc"] == 1.0

    def test_random(self):
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_scores = rng.uniform(0, 1, 100)
        result = auc_compute(y_true, y_scores)
        assert 0.2 <= result.extra["auc"] <= 0.8

    def test_alias(self):
        assert aucfn is auc_compute
