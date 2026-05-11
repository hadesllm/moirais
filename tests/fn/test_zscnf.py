"""Tests for morie.fn.zscnf -- Filled contour generation"""

import numpy as np
import pytest

from morie.fn.zscnf import contour_fill


class TestContourFill:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = contour_fill(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = contour_fill(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
