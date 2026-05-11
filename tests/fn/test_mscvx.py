"""Tests for morie.fn.mscvx -- 2D convex hull"""

import numpy as np
import pytest

from morie.fn.mscvx import convex_hull_2d


class TestConvexHull2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = convex_hull_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = convex_hull_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
