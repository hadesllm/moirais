"""Tests for morie.fn.zsrbt -- Thin plate spline RBF"""

import numpy as np
import pytest

from morie.fn.zsrbt import rbf_thinplate


class TestRbfThinplate:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rbf_thinplate(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = rbf_thinplate(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
