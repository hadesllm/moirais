"""Tests for morie.fn.svlin -- Linear spatial utility function"""

import numpy as np
import pytest

from morie.fn.svlin import linear_utility


class TestLinearUtility:
    def test_basic(self):
        x = np.array([1.0, 2.0])
        result = linear_utility(x, ideal_point=np.array([0.0, 0.0]))
        assert result.value is not None
        assert result.value >= 0

    def test_output_type(self):
        result = linear_utility(np.array([1.0, 2.0]))
        assert hasattr(result, "value")
