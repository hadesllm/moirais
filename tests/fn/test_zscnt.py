"""Tests for morie.fn.zscnt -- Contour line generation"""

import numpy as np
import pytest

from morie.fn.zscnt import contour_lines


class TestContourLines:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = contour_lines(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = contour_lines(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
