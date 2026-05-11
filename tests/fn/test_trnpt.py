"""Test turning_points_test (trnpt)."""
import numpy as np
import pytest

from morie.fn.trnpt import turning_points_test, trnpt
from morie.fn._containers import DescriptiveResult


class TestTurningPointsTest:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = turning_points_test(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "turning_points_test"

    def test_extra_keys(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = turning_points_test(x)
        assert "turning_points" in result.extra
        assert "expected" in result.extra
        assert "z_statistic" in result.extra
        assert "stationary" in result.extra

    def test_stationary_random(self):
        x = np.random.default_rng(42).standard_normal(500)
        result = turning_points_test(x)
        assert result.extra["stationary"] == True

    def test_alias(self):
        assert trnpt is turning_points_test
