"""Tests for morie.fn.zsvor -- Voronoi polygon areas"""

import numpy as np
import pytest

from morie.fn.zsvor import voronoi_areas


class TestVoronoiAreas:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = voronoi_areas(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = voronoi_areas(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
