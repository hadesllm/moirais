"""Test syndrome_compute."""
import numpy as np

from morie.fn._containers import DescriptiveResult
from morie.fn.syndc import syndc, syndrome_compute


class TestSyndromeCompute:
    def test_basic_zero_syndrome(self):
        H = np.array([
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        received = np.array([0, 0, 0, 0, 0, 0], dtype=np.uint8)
        result = syndrome_compute(H, received)
        assert isinstance(result, DescriptiveResult)
        assert result.extra["has_error"] is False
        assert result.extra["syndrome_weight"] == 0

    def test_value_equals_syndrome_sum(self):
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        received = np.array([0, 0, 0], dtype=np.uint8)
        result = syndrome_compute(H, received)
        assert result.value == 0.0

    def test_alias(self):
        assert syndc is syndrome_compute
