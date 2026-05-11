"""Tests for morie.fn.msvor -- 2D Voronoi diagram"""

import numpy as np
import pytest

from morie.fn.msvor import voronoi_2d


class TestVoronoi2d:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = voronoi_2d(data)
        assert result.value is not None

    def test_output_type(self):
        result = voronoi_2d(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
