"""Tests for morie.fn.zsdel -- Delaunay triangulation mesh"""

import numpy as np
import pytest

from morie.fn.zsdel import delaunay_mesh


class TestDelaunayMesh:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = delaunay_mesh(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = delaunay_mesh(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
