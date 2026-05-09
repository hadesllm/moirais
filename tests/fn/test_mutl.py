"""Tests for moirais.fn.mutl — mutual information."""
import numpy as np
import pytest
from moirais.fn.mutl import mutual_information


class TestMutualInfo:
    def test_identical(self):
        x = np.array([0, 0, 1, 1, 2, 2])
        res = mutual_information(x, x)
        assert res.estimate > 0

    def test_independent(self):
        rng = np.random.default_rng(42)
        x = rng.choice([0, 1], 1000)
        y = rng.choice([0, 1], 1000)
        res = mutual_information(x, y)
        assert res.estimate < 0.05
