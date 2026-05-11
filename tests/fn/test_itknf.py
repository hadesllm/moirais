"""Test itakura_dist (itknf)."""
import numpy as np
from morie.fn.itknf import itakura_dist, itknf
from morie.fn._containers import DescriptiveResult


class TestItknf:
    def test_basic(self):
        ar1 = [1.0, -0.5]
        ar2 = [1.0, -0.6]
        result = itakura_dist(ar1, 1.0, ar2, 1.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "itakura_distance"
        assert result.value >= 0

    def test_identical(self):
        ar = [1.0, -0.5, 0.2]
        result = itakura_dist(ar, 1.0, ar, 1.0)
        assert abs(result.value) < 1e-6

    def test_alias(self):
        assert itknf is itakura_dist
