"""Test superposition_test (supps)."""
import numpy as np

from morie.fn.supps import superposition_test, supps
from morie.fn._containers import DescriptiveResult


class TestSuperpositionTest:
    def test_lti_holds(self):
        h = np.array([1.0, 0.5, 0.25])
        rng = np.random.default_rng(42)
        x1 = rng.standard_normal(10)
        x2 = rng.standard_normal(10)
        result = superposition_test(h, x1, x2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "superposition_test"
        assert result.extra["holds"] is True

    def test_error_near_zero(self):
        h = np.array([1.0, -1.0])
        x1 = np.ones(5)
        x2 = np.arange(5, dtype=float)
        result = superposition_test(h, x1, x2)
        assert result.value < 1e-10

    def test_alias(self):
        assert supps is superposition_test
