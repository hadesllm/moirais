"""Tests for irtse -- SE(theta)."""
import numpy as np
from morie.fn.irtse import irt_se_theta
from morie.fn._containers import DescriptiveResult


class TestIrtSeTheta:
    def test_basic(self):
        params = {f"item_{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        result = irt_se_theta(params)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["se"]) == 81

    def test_se_finite(self):
        params = {f"item_{j}": {"a": 1.5, "b": 0.0} for j in range(10)}
        result = irt_se_theta(params)
        se = np.array(result.value["se"])
        assert np.all(np.isfinite(se))
        assert np.all(se > 0)
