"""Tests for moirais.fn.ptkde -- Spatial kernel density estimation"""

import numpy as np
import pytest

from moirais.fn.ptkde import spatial_kde


class TestSpatialKde:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spatial_kde(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = spatial_kde(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
