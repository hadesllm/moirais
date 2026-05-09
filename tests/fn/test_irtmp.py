"""Tests for irtmp -- MAP theta estimation."""
import numpy as np
from moirais.fn.irtmp import irt_map_theta
from moirais.fn._containers import DescriptiveResult


class TestIrtMapTheta:
    def test_basic(self):
        params = {f"item_{j}": {"a": 1.0, "b": float(j - 2)} for j in range(5)}
        responses = np.array([[1, 1, 0, 0, 0]])
        result = irt_map_theta(responses, params)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["theta"]) == 1

    def test_high_ability(self):
        params = {f"item_{j}": {"a": 1.0, "b": 0.0} for j in range(5)}
        responses = np.array([[1, 1, 1, 1, 1]])
        result = irt_map_theta(responses, params)
        assert result.value["theta"][0] > 0
