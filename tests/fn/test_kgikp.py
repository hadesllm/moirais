"""Tests for morie.fn.kgikp -- Indicator kriging probability"""

import numpy as np
import pytest

from morie.fn.kgikp import ik_probability


class TestIkProbability:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ik_probability(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = ik_probability(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
