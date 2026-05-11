"""Tests for morie.fn.msdel -- 2D Delaunay triangulation"""

import numpy as np
import pytest

from morie.fn.msdel import delaunay_2d


class TestDelaunay2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = delaunay_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = delaunay_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
