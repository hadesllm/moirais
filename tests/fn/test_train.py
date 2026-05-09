"""Tests for moirais.fn.train -- MMD distance."""

import numpy as np
from moirais.fn.train import mmd_distance, train
from moirais.fn._containers import DescriptiveResult


class TestTrain:
    __test__ = True

    def test_alias(self):
        assert train is mmd_distance

    def test_same_distribution(self):
        rng = np.random.default_rng(42)
        S = rng.normal(0, 1, (100, 3))
        T = rng.normal(0, 1, (100, 3))
        result = mmd_distance(S, T)
        assert isinstance(result, DescriptiveResult)
        assert result.value < 0.1

    def test_different_distributions(self):
        rng = np.random.default_rng(42)
        S = rng.normal(0, 1, (100, 3))
        T = rng.normal(5, 1, (100, 3))
        result = mmd_distance(S, T)
        assert result.value > 0.3
