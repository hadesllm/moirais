"""Tests for morie.fn.svmxu -- Mixed-norm spatial utility"""

import numpy as np
import pytest

from morie.fn.svmxu import mixed_utility


class TestMixedUtility:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = mixed_utility(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = mixed_utility(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
