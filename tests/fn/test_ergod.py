"""Test ergodicity_test (ergod)."""
import numpy as np
import pytest

from morie.fn.ergod import ergodicity_test, ergod
from morie.fn._containers import DescriptiveResult


class TestErgodicityTest:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        result = ergodicity_test(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ergodicity_test"

    def test_stationary_ergodic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        result = ergodicity_test(x, n_segments=10)
        assert result.extra["ergodic"] is True

    def test_too_short(self):
        with pytest.raises(ValueError):
            ergodicity_test([1.0, 2.0], n_segments=5)

    def test_alias(self):
        assert ergod is ergodicity_test
