"""Tests for morie.fn.svqud -- Quadratic spatial utility function"""

import numpy as np
import pytest

from morie.fn.svqud import quad_utility


class TestQuadUtility:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = quad_utility(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = quad_utility(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
